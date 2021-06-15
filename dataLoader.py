import os 

import torch
import pathlib
import re

from skimage import io, transform, exposure

import numpy as np
from torch.utils.data import Dataset, DataLoader
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
        
        label = img_name.replace(self.img_path, self.label_path).replace('.png', '.npy')
        landmarks = np.load(label)
        landmarks = landmarks.astype('float').reshape(-1, 2)
        
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h] #/ self.output_size

        return {'image': img, 'landmarks': landmarks}


class RandomScale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, margin):
        self.output_size = output_size
        self.margin = margin

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        variation = np.int(np.random.uniform(-self.margin, self.margin))
        outsize = self.output_size + variation
        
        h, w = image.shape[:2]
        if isinstance(outsize, int):
            if h > w:
                new_h, new_w = outsize * h / w, outsize
            else:
                new_h, new_w = outsize, outsize * w / h

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        
        if variation<0:
            v = abs(variation)
            img = np.pad(img, ((0, v), (0, v), (0, 0)), mode='constant', constant_values=0)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h] #/ self.output_size

        return {'image': img, 'landmarks': landmarks}
    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        if new_h == h or new_w == w:
            return {'image': image, 'landmarks': landmarks}
            
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
    
class AugColor(object):
    def __init__(self, gammaFactor):
        self.gammaf = gammaFactor

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        gamma = np.random.uniform(1 - self.gammaf, 1+self.gammaf)

        # Gamma
        image = exposure.adjust_gamma(image, gamma)

        return {'image': image, 'landmarks': landmarks}

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        angle = np.random.uniform(- self.angle, self.angle)

        image = transform.rotate(image, angle)
        
        centro = image.shape[0] / 2, image.shape[1] / 2
        
        landmarks -= centro
        
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        landmarks = np.dot(landmarks, R)
        
        landmarks += centro

        return {'image': image, 'landmarks': landmarks}


import cv2

def reverseVector(vector):
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    RCLAV = 23
    #LCLAV = 23
    
    p1 = RLUNG*2
    p2 = p1 + LLUNG*2
    p3 = p2 + HEART*2
    p4 = p3 + RCLAV*2
    
    rl = vector[:p1].reshape(-1,2)
    ll = vector[p1:p2].reshape(-1,2)
    h = vector[p2:p3].reshape(-1,2)
    rc = vector[p3:p4].reshape(-1,2)
    lc = vector[p4:].reshape(-1,2)
    
    return rl, ll, h, rc, lc

def drawBinary(img, organ, color):
    contorno = organ.reshape(-1, 1, 2)

    contorno = contorno.astype('int')
    
    img = cv2.drawContours(img, [contorno], -1, color, -1)
    
    return img

def getSeg(landmarks):
    leftlung, rightlung, heart, rc, lc = reverseVector(landmarks.reshape(-1))

    raw = np.zeros([512,512])
    
    raw = drawBinary(raw, leftlung, 1)
    raw = drawBinary(raw, rightlung, 1)
    
    raw = drawBinary(raw, heart, 2)
    
    raw = drawBinary(raw, rc, 3)
    raw = drawBinary(raw, lc, 3)
    
    return raw
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        raw = getSeg(landmarks)
        
        size = image.shape[0]
        image = image.transpose((2, 0, 1))
        landmarks = landmarks.reshape(-1, 2) / size
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float(),
                'seg': torch.unsqueeze(torch.from_numpy(raw),0).long()}