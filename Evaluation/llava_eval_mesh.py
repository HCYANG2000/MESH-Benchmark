import argparse
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm
import warnings
import os

from DataLoader.DataLoader import VideoFramesDataLoader

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    # Remove the video path and prompt argument and add the data path, data type, qa path arguments
    parser.add_argument("--data_path", type=str, help="Path to the data directory.", required=True)
    parser.add_argument("--data_type", type=str, help="Type of data to process.", required=True)
    parser.add_argument("--qa_path", type=str, help="Path to the QA file.", required=True)
    parser.add_argument("--input_image", type=str, help="Whether input the image data.", default='True')
    parser.add_argument("--fixed_length", type=str, help="Fixed video length around one given frame.", default=None)
    parser.add_argument("--sample_rate", type=int, help="Sample rate for video frames.", default=3)
    return parser.parse_args()

def custom_collate(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    answers = [item[2] for item in batch]
    qids = [item[3] for item in batch]
    return inputs, labels, answers, qids

# Preprocess the video qa data into input and answer
def preprocess(vision, ans_candidates, question, data_type, extra_args):
    # Preprocess the video data and prompt into input
    # Prompt is combination of question and answer candidates
    prompt_template_1 = "Question: {}\nOptions:"
    prompt_template_2 = "\nPlease select the correct answer ("
    prompt = prompt_template_1.format(question)
    for i in range(len(ans_candidates)):
        prompt += f"\n    {chr(65 + i)}. {ans_candidates[i]}"
    prompt += prompt_template_2
    for i in range(len(ans_candidates)):
        prompt += chr(65 + i)
        prompt += "/" if i < len(ans_candidates) - 1 else "):\n"

    if extra_args["input_image"] == 'True':
        # preprocess the video data
        preprocessor = extra_args['model_processor']
        tensor = preprocessor.preprocess(vision, return_tensors="pt")["pixel_values"].half()
    else:
        tensor = None

    if args.input_image == 'True':
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n" + prompt

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    tokenizer = extra_args['model_tokenizer']
            
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    image_sizes = [frame.numel() for frame in tensor]

    # Define the modality according to data type
    modalities = ["video"] * len(vision)

    # Return the preprocessed data
    return {'input_ids': input_ids, 'tensor': tensor, 'image_sizes': image_sizes, 'modalities': modalities}


def run_inference(args):
    """
    Run inference on MESH DataSet using the LLaVA-Video model.

    Args:
        args: Command-line arguments.
    """
    if args.fixed_length is not None:
        print('Since you specified fixed length, the for_get_frames_num will be ignored.')

    if args.sample_rate != 3:
        print('You are using a sample rate other than 3, which will only work for the action dataset.')
        print('Sample rate will only work if you specify fixed length as well.')
    
    # Initialize the model
    warnings.filterwarnings("ignore")
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, None, "llava_qwen", device_map=device_map, attn_implementation="sdpa")
    model.eval()

    # Create the output directory if it doesn't exist
    output_dir = None
    if args.fixed_length is not None:
        print('Since you specified fixed length, the for_get_frames_num will be ignored.')
        output_dir = os.path.join('work_dirs', args.model_path.split('/')[-1] + '-' + args.fixed_length, args.data_type)
        
    else:
        output_dir = os.path.join('work_dirs', args.model_path.split('/')[-1] + '-' + str(args.for_get_frames_num), args.data_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the output file name
    output_name = 'qa' + '_'+ args.qa_path.split('/')[-1][:-5]
    output_name += '_' + args.input_image if args.input_image == 'False' else ''

    # Set the output file path
    answers_file = os.path.join(output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")
    ans_file.write('[\n')

    # Load the QA data
    # The batch_size can only be 1
    # TODO: support batch inference
    data_loader = None
    batch_size = 1
    data_process_arguments = {
        'for_get_frames_num': args.for_get_frames_num, 
        'force_sample': args.force_sample, 
        'data_type': args.data_type, 
        'return_type': 'tensor', 
        'sample_rate': args.sample_rate, 
        'model_processor': image_processor, 
        'input_image': args.input_image,
        'fixed_length': args.fixed_length,
        'model_tokenizer': tokenizer,
        }
    data_loader_kwargs = {
        'annotation_path': args.qa_path,
        'data_path': args.data_path,
        'batch_size': batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': False,
        'preprocess': preprocess,
        'extra_arguments': data_process_arguments,
        'collate_fn': custom_collate
    }
    if args.data_type == "video" or args.data_type == "frames":
        data_loader = VideoFramesDataLoader(**data_loader_kwargs)
    else:
        raise ValueError("Data type not supported.")

    # import pdb;pdb.set_trace()
    for batch in tqdm(data_loader):
        inputs, labels, answers, qids = batch    
        input, label, answer, qid = inputs[0], labels[0], answers[0], qids[0]
        with torch.inference_mode():
            visions = [input['tensor'].to("cuda")]
            input_ids = input['input_ids'].to("cuda")
            output_ids = model.generate(
                input_ids,
                images=visions,
                image_sizes=input['image_sizes'],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                modalities=input['modalities'],
            )
        text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        sample_set = {'pred': text_outputs, 'label': label, 'answer': answer, 'id': qid}
        ans_file.write(json.dumps(sample_set) + ",\n")
        ans_file.flush()

    # Remove the last comma and close the file
    ans_file.seek(ans_file.tell() - 2)
    ans_file.write(']\n')
    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
