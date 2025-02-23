import argparse

import json
from vllm import LLM, SamplingParams
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from multiprocessing import freeze_support, set_start_method
import warnings
import os

from DataLoader.DataLoader import LNV_VideoFramesDataset

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
    visions_paths, answers, ans_candidates, labels, questions, data_types, data_path = examples[0:7]
    # Reshape the ans_candidates
    #ans_candidates = [[ans_candidates[i][j] for i in range(len(ans_candidates))] for j in range(len(ans_candidates[0]))]
    #print(visions_paths)
    #print(answers)
    #print(ans_candidates)
    #print(labels)
    #print(questions)
    #print(data_types)
    #print(data_path)
    #exit()

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
    return visions_paths, prompts, labels, answers, data_path


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

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
    # The batch size can only be 1
    # TODO: Allow batch inference
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

    if args.fixed_length:
        data_loader_kwargs['arguments']['fixed_length'] = args.fixed_length

    if args.data_type == "video" or args.data_type == "frames":
        data_loader = LNV_VideoFramesDataset(**data_loader_kwargs)
    else:
        raise ValueError("Data type not supported.")

    # import pdb;pdb.set_trace()
    for batch in data_loader:
        if batch == None:
            continue
        visions_paths, prompts, labels, answers, data_path = preprocess_data(batch)
        visions_paths = visions_paths if args.input_image == 'True' else None
        visions_path, prompt, label, answer, data_path = visions_paths[0], prompts[0], labels[0], answers[0], data_path[0]
        #print(len(visions_path))
        #print(prompt)
        #print(label)
        #print(answer)
        #print(data_path)
        #exit()
        sample_set = {"data_path": data_path, "A": answer, "label": label}
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": visions_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if args.system_prompt != "None":
            text = text.replace("You are a helpful assistant.", args.system_prompt)
        # Record the prompt
        sample_set["prompt"] = text
        # Process the vision info
        image_inputs, video_inputs = process_vision_info(messages)
        # Generate the final input
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        llm_inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
        }
        #print("length of input_ids: ", inputs.input_ids.shape)
        #print("shape of video inputs: ", inputs.pixel_values_videos.shape)
        #print("number of sampled frames: ", len(visions_paths))
        #print(f"Prompt: {text}")
        #print("--------------------")

        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        sample_set["pred"] = generated_text
        ans_file.write(json.dumps(sample_set) + ",\n")
        ans_file.flush()

    # Remove the last comma and close the file
    ans_file.seek(ans_file.tell() - 2)
    ans_file.write(']\n')
    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
