import torch
from .utils import load_video, load_frames

# Create a dataset for efficient loading of the data
class VideoFramesDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, answers, ans_candidates, data_types, labels, questions, qids, preprocess, extra1 = None, extra2 = None):
        # Initialize the variables
        self.data_paths = data_paths
        self.answers = answers
        self.ans_candidates = ans_candidates
        self.data_types = data_types
        self.labels = labels
        self.questions = questions
        self.extra1 = extra1
        self.extra2 = extra2
        self.preprocess = preprocess
        self.qids = qids

    def __len__(self):
        return len(self.answers)
    
    def __getitem__(self, idx):
        # Get the data
        answer = self.answers[idx]
        data_path = self.data_paths[idx]
        label = self.labels[idx]
        data_type = self.data_types[idx]
        qid = self.qids[idx]
        # Load the data based on the data type
        frames = None if self.extra2 is None else self.extra2[idx]
        # While loading the video and frames, the self.extra1 is used to pass the arguments and the self.extra2 is used to pass the frames
        if data_type == 'video':
            input = self.preprocess(load_video(data_path, self.extra1, frames), self.ans_candidates[idx], self.questions[idx], data_type, self.extra1)
        elif data_type == 'frames':
            input = self.preprocess(load_frames(data_path, self.extra1, frames), self.ans_candidates[idx], self.questions[idx], data_type, self.extra1)
        if input is None:
            raise ValueError('Data type not supported2')
        
        return input, label, answer, qid