# load Aria model & processor

import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import math
from tqdm import tqdm
import numpy as np
import sys
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','Qwen2VL')))

from DataLoaderQwen import Qwen_VideoFramesDataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def custom_collate(batch):
    try:
        visions_paths = [item[0] for item in batch]
        answers = [item[1] for item in batch]
        ans_candidates = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        questions = [item[4] for item in batch]
        data_types = [item[5] for item in batch]
        data_path = [item[6] for item in batch]
        return visions_paths, answers, ans_candidates, labels, questions, data_types, data_path
    except Exception as e:
        print(e)
        print('The items in batch:')
        for item in batch:
            print(item)
        return None

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

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
    #ans_candidates = [[ans_candidates[i][j] for i in range(len(ans_candidates))] for j in range(len(ans_candidates[0]))]
    #print(len(visions))
    #print(answers)
    #print(ans_candidates)
    #print(labels)
    #print(questions)
    #print(data_types)
    #print(data_path)
    #exit()

    # Because there is a list of videos or images, we need to separate them in a list
    #visions = [visions[i] for i in range(len(visions))]

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


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    if args.fixed_length is not None and 'final' not in args.qa_path:
        raise ValueError("Fixed length is only supported for final dataset.")
    if args.fixed_length is not None:
        print('Since you specified fixed length, the for_get_frames_num will be ignored.')
    if args.sample_rate != 3:
        print('You are using a sample rate other than 3, which will only work for the action dataset.')
        print('Sample rate will only work if you specify fixed length as well.')

    # Load the model and processor
    device_map = split_model(args.model_path.split('/')[-1])
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    # Create the output directory if it doesn't exist
    output_dir = None
    if args.fixed_length is not None:
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
        'arguments': {'for_get_frames_num': args.for_get_frames_num, 'force_sample': args.force_sample, 'data_type': args.data_type, 'return_type': 'path', 'sample_rate': args.sample_rate},
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
        data_loader = Qwen_VideoFramesDataLoader(**data_loader_kwargs)
    else:
        raise ValueError("Data type not supported.")

    transform = build_transform(input_size=448)
    
    for batch in tqdm(data_loader):
        if batch == None:
            continue
        visions, prompts, labels, answers, data_path = preprocess_data(batch)
        visions = visions if args.input_image == 'True' else None
        answers = [answers[i].item() if type(answers[i]) == torch.Tensor else answers[i] for i in range(len(answers))]
        visions, prompt, label, answer, data_path = visions[0], prompts[0], labels[0], answers[0], data_path[0]
        visions = torch.stack([transform(Image.open(vision)) for vision in visions]).to(torch.bfloat16).cuda()
        num_patches_list = [1 for _ in range(visions.shape[0])]
        #print(visions.shape)
        #print(prompts)
        #print(labels)
        #print(answers)
        #exit()
        sample_set = {"data_path": data_path, "A": answer, "label": label, "prompt": prompt}

        prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = prefix + prompt
        response = model.chat(tokenizer, visions, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=False)

        sample_set["pred"] = response
        ans_file.write( json.dumps(sample_set, ensure_ascii=False) + ',\n')
        ans_file.flush()

    ans_file.seek(ans_file.tell() - 2)
    ans_file.write(']')
    ans_file.close()