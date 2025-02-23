import torch
import math
from .utils import read_videoframes_qa_data
from .Dataset import LNV_VideoFramesDataset
    
# Dataloader to load video data
class LNV_VideoFramesDataLoader(torch.utils.data.DataLoader):
    def __init__(self,**kwargs):
        # Pop the data_path and annotation_path from the kwargs
        self.data_path = kwargs.pop('data_path')
        self.annotation_path = kwargs.pop('annotation_path')
        self.preprocessor = kwargs.pop('preprocessor', None)
        self.arguments = kwargs.pop('arguments')

        self.data = read_videoframes_qa_data(self.data_path, self.annotation_path, self.arguments['data_type'])

        self.data['extra1'] = self.arguments
        self.data['preprocessor'] = self.preprocessor
        # Create the dataset with the data
        self.dataset = LNV_VideoFramesDataset(**self.data)

        # Initialize the DataLoader
        super().__init__(self.dataset, **kwargs)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)