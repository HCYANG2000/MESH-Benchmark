import torch
from .utils import load_image, load_video, load_frames

# Create a dataset for efficient loading of the data
class UniDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, answers, ans_candidates, data_types, labels, questions, preprocessor, extra1 = None, extra2 = None):
        # Initialize the variables
        self.data_paths = data_paths
        self.answers = answers
        self.ans_candidates = ans_candidates
        self.data_types = data_types
        self.labels = labels
        self.questions = questions
        self.extra1 = extra1
        self.extra2 = extra2
        self.preprocessor = preprocessor

    def preprocess(self, data, data_type):
        if self.preprocessor is not None:
            if data_type == 'image' or data_type == 'video' or data_type == 'frames':
                data = self.preprocessor(images=data, return_tensors="pt").pixel_values
            elif data_type == 'text':
                data = self.preprocessor(text=data, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
            else:
                raise ValueError('Data type not supported1')
        return data
    
    def extra1_process(self, extra1, idx):
        return extra1[idx]
    
    def extra2_process(self, extra2, idx):
        return extra2[idx]

    def __len__(self):
        return len(self.answers)
    
    def __getitem__(self, idx):
        # Get the data
        answer = self.answers[idx]
        ans_candidate = self.preprocess(self.ans_candidates[idx], 'text')
        data_path = self.data_paths[idx]
        label = self.labels[idx]
        data_type = self.data_types[idx]
        question = self.questions[idx]
        # Load the data based on the data type
        data = None
        if data_type == 'image':
            data = self.preprocess(load_image(data_path), data_type)
        # While loading the video and frames, the self.extra1 is used to pass the arguments and the self.extra2 is used to pass the frames
        elif data_type == 'video':
            data = self.preprocess(load_video(data_path, self.extra1, self.extra2[idx]), data_type) if self.extra2 is not None else self.preprocess(load_video(data_path, self.extra1, None), data_type)
        elif data_type == 'frames':
            data = self.preprocess(load_frames(data_path, self.extra1, self.extra2[idx]), data_type) if self.extra2 is not None else self.preprocess(load_frames(data_path, self.extra1, None), data_type)
        if data is None:
            raise ValueError('Data type not supported2')
        
        if self.extra2 is not None:
            return data, answer, ans_candidate, label, question, data_type, data_path, self.extra1_process(self.extra1, idx), self.extra2_process(self.extra2, idx)
        elif self.extra1 is not None:
            return data, answer, ans_candidate, label, question, data_type, data_path, self.extra1_process(self.extra1, idx)
        else:
            return data, answer, ans_candidate, label, question, data_type, data_path

class LNV_Dataset(UniDataset):
    def __init__(self, data_paths, answers, ans_candidates, data_types, labels, questions, preprocessor, extra1 = None, extra2 = None):
        super().__init__(data_paths, answers, ans_candidates, data_types, labels, questions, preprocessor, extra1, extra2)

    # For LLaVA-like model, the image and video is preprocessed but the text is not
    def preprocess(self, data, data_type):
        if self.preprocessor is not None:
            if data_type == 'image' or data_type == 'video' or data_type == 'frames':
                try: 
                    # usually, video and frames can be processed like that
                    data = self.preprocessor.preprocess(data, return_tensors="pt")["pixel_values"].half()
                except Exception as e:
                    # But sometimes, the data is not in the right shape, especially for image
                    if "number of image dimensions: 2" in str(e):
                        data = self.preprocessor.preprocess(data[None], return_tensors="pt")["pixel_values"].half()
                        data = data[0]
                    # This is for clip, but now it should not be used
                    elif "'CLIPProcessor' object has no attribute 'preprocess'" in str(e):
                        data = self.preprocessor(images=data, return_tensors="pt").pixel_values.half()
            elif data_type == 'text':
                pass
            else:
                raise ValueError('Data type not supported3')
        return data


# Dataset to load video data
class LNV_VideoFramesDataset(LNV_Dataset):
    def __init__(self, data_paths, answers, ans_candidates, data_types, labels, questions, preprocessor, extra1 = None, extra2 = None):
        super().__init__(data_paths, answers, ans_candidates, data_types, labels, questions, preprocessor, extra1, extra2)

    # Overriding the process method since the extra1 store the arguments for loading the video
    def extra1_process(self, extra1, idx):
        # Just return itself
        return extra1
    
    # Overriding the process method since the extra2 store the frames
    def extra2_process(self, extra2, idx):
        # Just return the first frame
        return extra2[idx][0]