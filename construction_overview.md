
# **MESH Dataset - Construction Overview**
This section provides insight into how the datasets were constructed.

## **Motion Dataset**
Here are the steps involved in generating questions about the presence of different motions in video clips:
1. **Action Extraction**: Actions unrelated to conversation in video clips from the TVQA+ questions dataset are extracted, alongside conversations from the TVQA+ subtitle dataset, using the Deepseek tool. The extracted motion follows the format "A does B," where A and B represent individuals or objects in a succinct 10-word sentence.
2. **Name Replacement**: Names in the motion are substituted with the characteristics of the respective individuals, such as their clothing and gender.
3. **Event Hallucination Types**: By combining the person "A" and the motion "does B" from different sources, four different motion hallucination types are proposed:
    - **Event out of Video (EOV)**: Questions formulated by combining existing individuals from the video with events drawn from other videos.
    - **Person out of Video (POV)**: Questions formulated by individuals not present in the video and events existing within it.
    - **Similar Event (SE)**: Questions formulated by modifying existing video events. The altered event resembles the original but is not present in the video; for instance, transforming "open the door" into "closed the window.
    - **Event in Video (EIV)** and **Person in Video (PIV)**: Questions formulated by existing individuals and events from the video, but the individuals involved in each event are interchanged. For example, merging descriptions like "A woman wearing a blue sweater is cooking" and "A man wearing a gray t-shirt is sitting on the couch" to create scenarios where "A man wearing a gray t-shirt is cooking" and "A woman wearing a blue sweater is sitting on the couch."
4. **Question Generation**: Using both existing and non-existing actions, questions related to motion are formulated using the Deepseek tool.

## **Person Dataset**
Here are the steps involved in generating questions about the presence of different individuals in video clips:
1. **Feature Extraction**: GPT-4o-mini is employed to extract up to eight types of features of a person in a frame of a video clip. These features include:
    - Garment type: e.g., t-shirt, jacket, coat, button-up shirt, blouse, etc.
    - Garment color: e.g., red, blue, yellow, green, etc.
    - Gender: Male or female.
    - Glasses: Whether the person wears glasses or not.
    - Garment sleeve: e.g., long sleeves, short sleeves, no sleeves, and half sleeves.
    - Garment collar: Whether the upper garment has a collar or not.
    - Garment pocket: Whether the upper garment has pockets or not.
    - Garment Shade: Whether the overall color of the upper garment is dark or light.
2. **Human Verification**: The extracted features are checked in a human context.
3. **Feature Replacement**: The original features of a person in the video clip are substituted with dissimilar features to create a non-existing person.
4. **Question Formulation**: By utilizing both existing and non-existing persons, various types of questions related to individuals are generated using the Deepseek tool.

## **Object Dataset**
Here are the steps involved in generating questions about the presence of different obejects in video clips:
1. **Object Extraction**: Objects in the video clips are extracted using TVQA+ bounding box dataset with the Deepseek tool.
2. **Scene Location Extraction**: The location of the scene in the video clips is extracted using the Large Language-and-Vision Assistant model.
3. **Similar Object Generation**: By utilizing the extracted objects and scene locations, objects with similar shapes to the original ones that do not exist in the given place are generated using Deepseek, acting as non-existing objects.
4. **Question Formulation**: Both existing and non-existing objects are employed to formulate various types of questions related to objects using the Deepseek tool.
