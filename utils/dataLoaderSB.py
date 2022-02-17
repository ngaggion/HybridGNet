import os 

import torch
import pathlib
import re

from skimage import io

import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


class LandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_path, label_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
                
        data_root = pathlib.Path(img_path)
        all_files = list(data_root.glob('*.png'))
        all_files = [str(path) for path in all_files]
        all_files.sort(key = natural_key)
        
        self.images = all_files
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        image = io.imread(img_name).astype('float') / 255.0
        image = np.expand_dims(image, axis=2)
        
        label = img_name.replace(self.img_path, self.label_path)
        label = io.imread(label).astype('bool')
        label = np.expand_dims(label, axis=2)
        
        sample = {'image': image, 'seg': label}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['seg']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image).float(),
                'seg': torch.from_numpy(label).bool()}