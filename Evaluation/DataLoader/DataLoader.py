import torch
import math
from .utils import read_videoframes_qa_data
from .Dataset import VideoFramesDataset
    
# Dataloader to load video data
class VideoFramesDataLoader(torch.utils.data.DataLoader):
    def __init__(self,**kwargs):
        # Pop the data_path and annotation_path from the kwargs
        self.data_path = kwargs.pop('data_path')
        self.annotation_path = kwargs.pop('annotation_path')
        self.preprocess = kwargs.pop('preprocess')
        self.extra_arguments = kwargs.pop('extra_arguments')

        # Read the data
        self.data = read_videoframes_qa_data(self.data_path, self.annotation_path, self.extra_arguments['data_type'])

        # Record the extra arguments and preprocessor
        self.data['extra1'] = self.extra_arguments
        self.data['preprocess'] = self.preprocess

        # Create the dataset with the data
        self.dataset = VideoFramesDataset(**self.data)

        # Initialize the DataLoader
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)