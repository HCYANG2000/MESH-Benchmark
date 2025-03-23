import argparse

import json
from vllm import LLM, SamplingParams
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from multiprocessing import freeze_support, set_start_method
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
    parser.add_argument("--tokens_per_image", type=int, default=None)
    # Remove the video path and prompt argument and add the data path, data type, qa path arguments
    parser.add_argument("--data_path", type=str, help="Path to the data directory.", required=True)
    parser.add_argument("--data_type", type=str, help="Type of data to process.", required=True)
    parser.add_argument("--qa_path", type=str, help="Path to the QA file.", required=True)
    parser.add_argument("--input_image", type=str, help="Whether input the image data.", default='True')
    parser.add_argument("--system_prompt", type=str, help="The system prompt of input.", default="None")
    parser.add_argument("--tensor_parallel_size", type=int, help="Number of gpu used in tensor parallel", default=2)
    parser.add_argument("--fixed_length", type=str, help="Fixed video length.", default=None)
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
    """
    Preprocess the data for the model.

    Args:
        examples: The examples to preprocess.

    Returns:
        The preprocessed examples.
    """

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

    def calculate_fps(for_get_frames_num, fixed_length, sample_rate):
        if fixed_length:
            return 3 / sample_rate
        return 60 / for_get_frames_num
    
    if extra_args['system_prompt'] != "None":
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text":extra_args['system_prompt']},
                ],
            },
        ]
    else:
        messages = []
    
    if extra_args['input_image'] == 'True':
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": paths,
                        "fps": calculate_fps(extra_args['for_get_frames_num'], extra_args['fixed_length'], extra_args['sample_rate']),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        )

    processor = extra_args['model_processor']

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process the vision info
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    # Generate the final input
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    inputs = {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    # Return the preprocessed data
    return inputs


def run_inference(args):
    """
    Run inference on MESH DataSet using the Qwen2.5VL model.

    Args:
        args: Command-line arguments.
    """
    if args.fixed_length is not None:
        print('Since you specified fixed length, the for_get_frames_num argument will be ignored.')

    if args.sample_rate != 3:
        print('You are using a sample rate other than 3, which will only work for the action dataset.')
        print('Sample rate will only work if you specify fixed length as well.')

    # Initialize the model
    warnings.filterwarnings("ignore")
    freeze_support()
    set_start_method('spawn')
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        download_dir=os.getenv("HF_HOME"),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len = 25856, # Adjust the max model length accordingly
    )    

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )

    # Init the processor
    if args.tokens_per_image:
        min_pixels = args.tokens_per_image*28*28
        max_pixels = args.tokens_per_image*28*28
        processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    else:
        processor = AutoProcessor.from_pretrained(args.model_path)

    # Create the output directory if it doesn't exist
    output_dir = None
    if args.fixed_length:
        # if the fixed length is set, then the for_get_frames_num will be ignored
        output_dir = os.path.join('work_dirs', args.model_path.split('/')[-1] + '-' + str(args.fixed_length), args.data_type)
    else:  
        output_dir = os.path.join('work_dirs', args.model_path.split('/')[-1] + '-' + str(args.for_get_frames_num), args.data_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the output file name
    output_name = 'qa' + '_'+ args.qa_path.split('/')[-1][:-5]
    output_name += '_system' if args.system_prompt != "None" else ''
    output_name += '_' + args.input_image if args.input_image == 'False' else ''

    # Set the output file path
    answers_file = os.path.join(output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")
    ans_file.write('[\n')

    # Load the QA data
    data_loader = None
    batch_size = 10
    data_process_arguments = {
        'for_get_frames_num': args.for_get_frames_num, 
        'force_sample': args.force_sample, 
        'data_type': args.data_type, 
        'return_type': 'path', 
        'sample_rate': args.sample_rate, 
        'model_processor': processor, 
        'input_image': args.input_image,
        'fixed_length': args.fixed_length,
        'system_prompt': args.system_prompt
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
    for batch in data_loader:
        inputs, labels, answers, qids = batch    
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        generated_texts = [outputs[i].outputs[0].text for i in range(len(outputs))]
        for i in range(len(generated_texts)):
            ans_file.write(json.dumps({'pred': generated_texts[i], 'label': labels[i], 'answer': answers[i], 'qid': qids[i]}) + ",\n")
        ans_file.flush()

    # Remove the last comma and close the file
    ans_file.seek(ans_file.tell() - 2)
    ans_file.write(']\n')
    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
