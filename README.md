
# **MESH Dataset - Download and Description**
## **Dataset**
The MESH dataset is a large collection of questions and answers that inquire about the presence of different elements in video clips from the television series "The Big Bang Theory." Drawing data from the TVQA+ dataset, it generates over 140,000 questions that revolve around identifying Motion, Person, and Object. These questions are organized into corresponding folders named "Motion," "Person," and "Object" for clarity and ease of access.

### **Motion Folder**

The "Motion" folder contains questions related to the motion depicted in the video clips. Within this folder, there are four JSON files:
| File| Description|
|--|--|
|action|  This file contains questions related to the motion depicted in the video clip. Questions about sound-related motion are not included, and all questions are in a yes-no format.|
|talk|This file includes questions about the conversations happening in the video clip, all presented in a yes-no format.|
|mc_action|This file holds questions about the motion shown in the video clip. Questions concerning sound-related motion are excluded, with each question offering four answer choices.|
|mc_talk|This file features questions about the conversations taking place in the video clip, each question providing four answer choices.|

### **Person Folder**
The "Person" folder contains questions related to the person depicted in the video clips. Within this folder, the JSON files are formatted as **{granularity}{answer_choices}.json**.

**{granularity}** indicates the number of target person features included in each question within the file. There are four distinct types of granularity:
| Granularity| Description|
|--|--|
|raw_grain|In this type of JSON file, three features of the target person are included in the questions: Garment color, Garment type, and Gender.|
|medium_grain|  In this type of JSON file, five features of the target person are included in the questions: Garment color, Garment type, Gender, Garment sleeve, and Glasses.|
|mixed_grain|In this type of JSON file, five features of the target person are included in the questions, with the feature being randomly selected from the extracted features.|
|fine_grain|In this type of JSON file, all extracted features are included in the questions, with a maximum of eight features being included.|

**{answer_choices}** serves as an indicator for the question type as follows:

1. If this part is empty, the questions are in a yes-no format.
2. If it is **_four**, the questions provide four answer choices.



### **Object Folder**

The "Object" folder contains questions related to the Object depicted in the video clips. Within this folder, there are two JSON files:
| File| Description|
|--|--|
|object|This file includes questions related to the object depicted in the video clip, all presented in a yes-no format.|
|mc_object|This file includes questions related to the object depicted in the video clip, each question providing four answer choices.|

## **Annotations**
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
