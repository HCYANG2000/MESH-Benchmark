import os
import torch
from tqdm import tqdm
import sys
import argparse
import json

from lmdeploy import pipeline
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from lmdeploy import TurbomindEngineConfig
from lmdeploy.vl import load_image

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
    parser.add_argument("--tensor_parallel_size", type=int, help="Number of gpu used in tensor parallel", default=1)
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
def preprocess(paths, ans_candidates, question, data_type, extra_args):
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
        visions = [preprocessor(path) for path in paths]
        prefix = ''.join([f'Frame{i+1}: {IMAGE_TOKEN}\n' for i in range(len(visions))])
    else:
        visions = []

    question = prefix + prompt

    content = [{'type': 'text', 'text': question}]
    for encoded_img in visions:
        content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encoded_img}'}})

    messages = [
        {
            'role': 'user',
            'content': content
        }
    ]

    return {"messages": messages}

def load_and_encode_img(image_path):
    image = load_image(image_path)
    encoded_image = encode_image_base64(image)
    return encoded_image

if __name__ == "__main__":
    """
    Run inference on MESH DataSet using the InternVL-2.5 model with huggingface transformers implementation.

    Args:
        args: Command-line arguments.
    """
    args = parse_args()
    if args.fixed_length is not None:
        print('Since you specified fixed length, the for_get_frames_num will be ignored.')

    if args.sample_rate != 3:
        print('You are using a sample rate other than 3, which will only work for the action dataset.')
        print('Sample rate will only work if you specify fixed length as well.')

    # Load the model
    pipe = pipeline(args.model_path, backend_config=TurbomindEngineConfig(session_len=32768, tp=args.tensor_parallel_size), log_level = 'WARNING', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True,)

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
    # The batch_size can only be 1!!!
    # TODO: support batch inference
    data_loader = None
    batch_size = 1
    data_process_arguments = {
        'for_get_frames_num': args.for_get_frames_num, 
        'force_sample': args.force_sample, 
        'data_type': args.data_type, 
        'return_type': 'path', 
        'sample_rate': args.sample_rate, 
        'model_processor': load_and_encode_img,
        'input_image': args.input_image,
        'fixed_length': args.fixed_length,
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
        'collate_fn': custom_collate,
    }

    if args.data_type == "video" or args.data_type == "frames":
        data_loader = VideoFramesDataLoader(**data_loader_kwargs)
    else:
        raise ValueError("Data type not supported.")
    
    for batch in tqdm(data_loader):
        inputs, labels, answers, qids = batch
        input, label, answer, qid = inputs[0], labels[0], answers[0], qids[0]
        messages = input["messages"]
        response = pipe(messages, top_k=1)
        sample_set = {'pred': response, 'label': label, 'answer': answer, 'id': qid}
        ans_file.write(json.dumps(sample_set) + ",\n")
        ans_file.flush()

    ans_file.seek(ans_file.tell() - 2)
    ans_file.write(']')
    ans_file.close()