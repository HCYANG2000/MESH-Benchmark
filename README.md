
# **MESH Dataset - Download and Description**

## Benchmark description
The MESH dataset is a large collection of questions and answers (QAs) that inquire about the presence of hallucination phenomenons in utilizing large vision-language models (LVLMs) in understanding videos.

Drawing data from the TVQA+ dataset (https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa_plus.html), it generates over 140,000 questions that revolve around identifying three key aspects of human understanding videos: *Setting (objects)*, *Character* and *Stage (action and dialogues)*. We also organize the QA set in these aspects.

## Annotations
Each JSON file contains a list of dicts, each dict has the following entries:
| Key| Type  | Description|
|--|--|--|
|  answer |int  | The index of the answer in the ans_candidates array. For instance, if answer is 2, then the third element in ans_candidates corresponds to the answer.|
|  video_name |str  | The name of the video clip that accompanies the question. Video names follow the format 's{season_number}e{episode_number}\_seg{segment_number}\_clip\_{clip_number}'. For example, 's06e12_seg02_clip_16' indicates a video clip from season 6, episode 12 of the TV show, which is the 16th clip of the 2nd segment. |
|  ans_candidates |list  | Contains the potential answers to the question.|
|  question |str  | The question being asked.|
|  label |str  | In setting and character dataset, this field can be omitted. In the stage dataset, this field denote different type of stage and is useful during evaulation (check our paper for stage types).|
|  focus |str  | A field considered not useful.|
|  frames |list | Frame annotations associated with the question. For stage dataset, [[0, 5]] denotes approximately these frames are required to answer the question. For setting and character, the annotation [80, 20, ...] signifies frame 80, 20 and ... are the frames that target object or character appears.|

## Benchmark Categories

### Setting

*Setting* contains questions related to the presence of objects depicted in the video clips. This category is called "Setting" because each object in question is closely associated with the setting of the video (check our paper for more detail). Within this folder, there are two JSON files:

| File      | Description                                                  | 
| --------- | ------------------------------------------------------------ | 
| object    | This file includes questions related to the object depicted in the video clip, all presented in a yes-no format. | 
| mc_object | This file includes questions related to the object depicted in the video clip, each question providing four answer choices. |

A sample of the QA from ```mc_object.json``` is shown below:

```json
{
    "answer": "C",
    "video_name": "s03e09_seg02_clip_02",
    "question": "What is the object present in the video?",
    "ans_candidates": [
        "Birdbath presents in the video.",
        "Seat belt presents in the video.",
        "Lunchroom presents in the video.",
        "Appointment book presents in the video."
    ],
    "label": "o",
    "focus": "lunchroom",
    "frames": [
        1,
        13,
        7
    ]
},
```

### Character
*Character* contains questions related to the persons in the video and their appearances, such as cloth type and color, depicted in the video clips. 

Within this folder, we provide four granularity for both *yes-no* and *mc* (multiple-choice) format. **Granularity** indicates the number of target person features included in each question within the file. 

| Granularity  | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| coarse_grain    | In this type of JSON file, three features of the target person are included in the questions: Garment color, Garment type, and Gender. |
| medium_grain | In this type of JSON file, five features of the target person are included in the questions: Garment color, Garment type, Gender, Garment sleeve, and Glasses. |
| mixed_grain  | In this type of JSON file, five features of the target person are included in the questions, with the feature being randomly selected from the extracted features. |
| fine_grain   | In this type of JSON file, all extracted features are included in the questions, with a maximum of eight features being included. |

For example, ```fine_grain.json``` contains questions in a fine granularity with *yes-no* format while ```fine_grain_four.json``` contains that with *multiple-choices* format.

A sample of the QA from ```coarse_grain.json``` is shown below:

```json
{
    "answer": 0,
    "video_name": "s01e01_seg01_clip_01",
    "ans_candidates": [
        "a man wearing jacket appears in the video.",
        "a man wearing jacket does not appear in the video."
    ],
    "question": "Does a man wearing jacket appear in the video?",
    "label": "p",
    "focus": "N/A",
    "frames": [
        "80"
    ]
}
```

### Stage

*Stage* contains questions related to the action and talking of characters, which corresponds to "Stage (Dialogue)" and "Stage (Non-Dialogue)" in the paper. Notice that because we purly focus on video content, any question concerning content of converstation are excluded, and the "talking" merely involve the "talking" behavior. The detailed division is listed below

| File| Description|
|--|--|
|action|  This file contains questions related to the action depicted in the video clip. Questions about dialogue-related action are not included, and all questions are in a yes-no format.|
|talk|This file includes questions about the "talking" behavior in the video clip, all presented in a yes-no format.|
|mc_action|This file holds questions about the action shown in the video clip. Each question offering four answer choices.|
|mc_talk|This file features questions about the "talking" behavior taking place in the video clip, each question providing four answer choices.|

A sample of the QA from ```action.json``` is shown below:

```json
{
    "answer": 1,
    "video_name": "s04e09_seg02_clip_05",
    "ans_candidates": [
        "The video depicts a woman wearing a red sweater using chopsticks.",
        "The video does not depict a woman wearing a red sweater using chopsticks."
    ],
    "question": "Does the video show a woman wearing a red sweater using chopsticks?",
    "label": "wrong_out_video_action",
    "focus": "N/A",
    "frames": [
        [
            53,
            71
        ]
    ]
},
```

A sample of the QA from ```mc_talk.json``` is shown below:

```json
{
    "answer": "D",
    "video_name": "s03e18_seg02_clip_01",
    "ans_candidates": [
        "A man in a blue robe was speaking in the video.",
        "A woman in a white polo shirt was speaking in the video.",
        "A man in a green robe was speaking in the video.",
        "A man in a gray sweater was speaking in the video."
    ],
    "question": "Who was the speaker in the video clip?",
    "label": "wrong_out_vid_person",
    "focus": "N/A",
    "frames": [
        [
            54,
            72
        ]
    ]
},
```


## Benchmark Usage

### Data Preparation

Evaluation of model requires qa pairs and video data. The qa pairs have been placed in ```data/QA``` according to their categories. To download the video data, you need to send application according to the instruction in official website of TVQA dataset (https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa_plus.html) to get the link. After you download the TVQA video, you can find a subfolder call ```bbt_frames```, which contains the required video frames. Finally, rename the ```bbt_frames``` as ```raw_videos``` and move it to ```data/```. The final files structure should be like:

```
MESH-Benchmark/
|–– Evaluation/
|   |-- DataLoader/
|   |-- qwen_eval_mesh.py
|   |-- ...
|–– data/
|   |–– raw_videos/
|       |-- s01e01_seg01_clip_00/
|           |-- 00001.jpg
|           |-- 00002.jpg
|           |-- 00003.jpg
|           |-- ...
|       |-- s01e01_seg01_clip_01/
|       |-- s01e01_seg01_clip_02/
|       |-- ...
|   |-- QA/
|       |-- Character/
|           |-- coarse_grain_four.json
|           |-- coarse_grain.json
|           |-- ...
|       |-- Setting/
|           |-- object.json
|           |-- mc_object.json
|       |-- Stages/
|           |-- action.json
|           |-- mc_action.json
|           |-- ...
|–– README.md
|–– ...
```

### Arguments

All the commands below following fixed format like:

```
python ./Evaluation/[EvaluationScript].py --model-path [ModelPath] --for_get_frames_num [UniformlySampledFramesNumber] --force_sample [ForceSampleIfThereIsNoEnoughFrames] --data_path data/raw_videos/ --qa_path data/QA/[PathToQAPairs] --data_type frames --fixed_length [FixedLengthType]
```

**EvaluationScript**: Different models or different implementation have different python script, for example ```qwen_eval_mesh_vllm``` is for Qwen2.5VL with vllm implementation. Please refer to next section for more detail and choose the python script accordingly.

**ModelPath**: Model's path in huggingace or local directory, for example ```Qwen/Qwen2.5-VL-7B-Instruct```.

**UniformlySampledFramesNumber**: Number of frames that are uniformly sampled from one video, for example, ```32```. If the *FixedLengthType* is empty, then the script will uniformly sample frames from video according to this argument. If the *FixedLengthType* is not empty, then this argument will be ignored.

**ForceSampleIfThereIsNoEnoughFrames**: When the *FixedLengthType* is empty and *UniformlySampledFramesNumber* exceeds the total number of frames in the video, if this argument is ```True```, then the sample frames will be duplicated to guarantee the sampled frames number is equal to *UniformlySampledFramesNumber*, othervise, the script just returns all the frames in the video.

**PathToQAPairs**: Path to the QA set for evaluation, for example, ```Character/coarse_grain.json```. Different QA sets and their feature are introduced in the *Annotations* Section. You can choose any QA sets that you are interested in.

**FixedLengthType**: 

### Qwen2.5VL

Following the instruction in Qwen2.5VL official repository (https://github.com/QwenLM/Qwen2.5-VL) to install the conda environment. Then, run the following command to evaluate Qwen2.5VL 7B model using huggingface transformer:

```
python ./Evaluation/qwen_eval_mesh.py --model-path Qwen/Qwen2.5-VL-7B-Instruct --for_get_frames_num 64 --force_sample True --data_path data/raw_videos/ --qa_path data/QA/Character/coarse_grain.json --data_type frames --fixed_length short
```

If you prefer vllm, 

```
python ./Evaluation/qwen_eval_mesh_vllm.py --model-path Qwen/Qwen2.5-VL-7B-Instruct --for_get_frames_num 64 --force_sample True --data_path data/raw_videos/ --qa_path data/QA/Character/coarse_grain.json --data_type frames --fixed_length short --tensor_parallel_size 1
```