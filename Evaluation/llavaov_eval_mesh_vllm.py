import argparse
import torch
from torch.utils.data._utils.collate import default_collate

from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset
from multiprocessing import freeze_support, set_start_method
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm

import warnings
import sys
import os

# Add the directory two levels up to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..','LLaVA-Next')))

from DataLoader import LNV_ImageDataLoader, LNV_VideoFramesDataLoader

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
    try:
        return default_collate(batch)
    except Exception as e:
        print(e)
        print('The items in batch:')
        for item in batch:
            print(item)
        return None

# Preprocess the video qa data into input and answer
def preprocess_data(examples):
    """
    Preprocess the data for the model.

    Args:
        examples: The examples to preprocess.

    Returns:
        The preprocessed examples.
    """
    # Extract the label, answer, question, video_name and ans_candidates from the examples
    visions, answers, ans_candidates, labels, questions, data_types, data_path = examples[0:7]
    # Reshape the ans_candidates
    ans_candidates = [[ans_candidates[i][j] for i in range(len(ans_candidates))] for j in range(len(ans_candidates[0]))]
    #print(visions.shape)
    #print(answers)
    #print(ans_candidates)
    #print(labels)
    #print(questions)
    #print(data_types)
    #print(data_path)
    #exit()

    # Because there is a list of videos or images, we need to separate them in a list
    visions = [visions[i] for i in range(len(visions))]

    # Preprocess the data as {'prompt': '...', 'label': '...', 'answer': '...', 'video_name': '...'}
    # Prompt is combination of question and answer candidates
    prompts = []
    prompt_template_1 = "Question: {}\nOptions:"
    prompt_template_2 = "\nPlease select the correct answer ("
    for i in range(len(questions)):
        prompt_i = prompt_template_1.format(questions[i])
        for j in range(len(ans_candidates[i])):
            prompt_i += f"\n    {chr(65 + j)}. {ans_candidates[i][j]}"
        prompt_i += prompt_template_2
        for j in range(len(ans_candidates[i])):
            prompt_i += chr(65 + j)
            prompt_i += "/" if j < len(ans_candidates[i]) - 1 else "):\n"
        prompts.append(prompt_i)

    # Return the preprocessed data
    return visions, prompts, labels, answers, data_path


def run_inference(args):
    """
    Run inference on MESH DataSet using the LLaVA model.

    Args:
        args: Command-line arguments.
    """
    freeze_support()
    set_start_method('spawn')
    if args.fixed_length is not None and 'final' not in args.qa_path:
        raise ValueError("Fixed length is only supported for final dataset.")
    
    if args.fixed_length is not None:
        print('Since you specified fixed length, the for_get_frames_num will be ignored.')

    if args.sample_rate != 3:
        print('You are using a sample rate other than 3, which will only work for the action dataset.')
        print('Sample rate will only work if you specify fixed length as well.')

    # Initialize the model
    warnings.filterwarnings("ignore")
    llm = LLM(
        model=args.model_path,
        max_model_len=32768,
        tensor_parallel_size=4,
    )

    sampling_params = SamplingParams(temperature=0,
                                    max_tokens=64,
                                    stop_token_ids=None)

    # Create the output directory if it doesn't exist
    output_dir = None
    if args.fixed_length is not None:
        print('Since you specified fixed length, the for_get_frames_num will be ignored.')
        output_dir = os.path.join('work_dirs', args.model_path.split('/')[-1] + '-' + args.fixed_length, args.data_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
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
    data_loader_kwargs = {
        'annotation_path': args.qa_path,
        'data_path': args.data_path,
        'arguments': {'for_get_frames_num': args.for_get_frames_num, 'force_sample': args.force_sample, 'data_type': args.data_type, 'sample_rate': args.sample_rate},
        'batch_size': batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': False,
        'preprocessor': None,
        'collate_fn': custom_collate,
    }

    if args.fixed_length is not None:
        data_loader_kwargs['arguments']['fixed_length'] = args.fixed_length

    if args.data_type == "video" or args.data_type == "frames":
        data_loader = LNV_VideoFramesDataLoader(**data_loader_kwargs)
    elif args.data_type == "image":
        data_loader = LNV_ImageDataLoader(**data_loader_kwargs)
    else:
        raise ValueError("Data type not supported.")

    # import pdb;pdb.set_trace()
    for batch in tqdm(data_loader):
        if batch == None:
            continue
        visions, prompts, labels, answers, data_path = preprocess_data(batch)
        visions = [vision.numpy() for vision in visions] if args.input_image == 'True' else None
        answers = [answers[i].item() if type(answers[i]) == torch.Tensor else answers[i] for i in range(len(answers))]
        #print(visions[0].shape)
        #print(prompts)
        #print(labels)
        #print(answers)
        #exit()

        sample_set_list = [{"Q": prompts[i], "data_path": data_path[i], "A": answers[i], "label": labels[i]} for i in range(len(prompts))]
        question_list = prompts

        # try:
        # Run inference on the video and add the output to the list
        qs_list = question_list
        if args.input_image == 'True':
            prompt_list = [f"<|im_start|>user\n<video>\n{qs}<|im_end|>\n<|im_start|>assistant\n" for qs in qs_list]
        else:
            prompt_list = [f"<|im_start|>user\n{qs}<|im_end|>\n<|im_start|>assistant\n" for qs in qs_list]

        # Record the prompt
        for i in range(len(sample_set_list)):
            sample_set_list[i]["prompt"] = prompt_list[i]
            #print(f"Prompt: {prompt_list[i]}")
        modalty = "video" if args.data_type == "video" or "frames" else "image"
        inputs = {
            "prompt": prompt_list[0],
            "multi_modal_data": {
                modalty: visions[0]
            },
        }

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        # Record the output
        for i, sample_set in enumerate(sample_set_list):
            sample_set["pred"] = outputs[i].outputs[0].text
            ans_file.write(json.dumps(sample_set, ensure_ascii=False) + ",\n")
            ans_file.flush()

    # Remove the last comma and close the file
    ans_file.seek(ans_file.tell() - 2)
    ans_file.write(']\n')
    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
