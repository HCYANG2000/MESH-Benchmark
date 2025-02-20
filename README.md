
# **MESH Dataset - Download and Description**
## Benchmark description
The MESH dataset is a large collection of questions and answers (QAs) that inquire about the presence of hallucination phenomenons in utilizing large vision-language models (LVLMs) in understanding videos.

Drawing data from the TVQA+ dataset (https://nlp.cs.unc.edu/data/jielei/tvqa/tvqa_public_html/download_tvqa_plus.html), it generates over 140,000 questions that revolve around identifying three key aspects of human understanding videos: *Setting (objects)*, *Character Appearance* and *Stage (motions)*. We also organize the QA set in these aspects.

### Setting

*Setting* contains questions related to the presence of objects depicted in the video clips. Within this folder, there are two JSON files:

| File      | Description                                                  |      |
| --------- | ------------------------------------------------------------ | ---- |
| object    | This file includes questions related to the object depicted in the video clip, all presented in a yes-no format. |      |
| mc_object | This file includes questions related to the object depicted in the video clip, each question providing four answer choices. |      |

### Character Appearance
*Character Appearance* contains questions related to the persons and their appearances, such as cloth type and color, depicted in the video clips. 

Within this folder, we provide four granularity for both *yes-no* and *mc* (multiple-choice) format. **Granularity** indicates the number of target person features included in each question within the file. 

| Granularity  | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| raw_grain    | In this type of JSON file, three features of the target person are included in the questions: Garment color, Garment type, and Gender. |
| medium_grain | In this type of JSON file, five features of the target person are included in the questions: Garment color, Garment type, Gender, Garment sleeve, and Glasses. |
| mixed_grain  | In this type of JSON file, five features of the target person are included in the questions, with the feature being randomly selected from the extracted features. |
| fine_grain   | In this type of JSON file, all extracted features are included in the questions, with a maximum of eight features being included. |

For example, <u>fine_grain.json</u> contains questions in a fine granularity with *yes-no* format while <u>fine_grain_four.json</u> contains that with *multiple-choices* format.

### Stage

*Stage* contains questions related to the motion depicted in the video clips. We separate the motions containing in the videos into two aspects: talking and other actions. The detailed division is listed below

| File| Description|
|--|--|
|action|  This file contains questions related to the motion depicted in the video clip. Questions about sound-related motion are not included, and all questions are in a yes-no format.|
|talk|This file includes questions about the conversations happening in the video clip, all presented in a yes-no format.|
|mc_action|This file holds questions about the motion shown in the video clip. Questions concerning sound-related motion are excluded, with each question offering four answer choices.|
|mc_talk|This file features questions about the conversations taking place in the video clip, each question providing four answer choices.|


## Benchmark Annotations
Each JSON file contains a list of dicts, each dict has the following entries:
| Key| Type  | Description|
|--|--|--|
|  answer |int  | The index of the answer in the ans_candidates array. For instance, if answer is 2, then the third element in ans_candidates corresponds to the answer.|
|  video_name |str  | The name of the video clip that accompanies the question. Video names follow the format '{show_name_abbr}_s{season_number}e{episode_number}_seg{segment_number}clip{clip_number}'. For example, 'friends_s06e12_seg02_clip_16' indicates a video clip from season 6, episode 12 of the TV show 'Friends', which is the 16th clip of the 2nd segment. Note that video clips for 'The Big Bang Theory' exclude '{show_name_abbr}' in their names.|
|  ans_candidates |list  | Contains the potential answers to the question.|
|  question |str  | The question being asked.|
|  label |str  | A field considered not useful.|
|  focus |str  | A field considered not useful.|
|  frames |list | Frame annotations associated with the question. For example, [0, 5] denotes frames starting at 0 and ending at 5. The annotation ["80"] signifies frame 80 in relation to the question.|

A sample of the QA is shown below:
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

## Benchmark Usage

... To be finished
