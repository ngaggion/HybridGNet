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

        self.label_path = label_path
        self.transform = transform
                
        data_root = pathlib.Path(self.label_path)
        all_files = list(data_root.glob('*.npy'))
        all_files = [str(path) for path in all_files]
        all_files.sort(key = natural_key)
        
        self.images = all_files
        print('Total of landmarks:', len(all_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.images[idx]
        landmarks = np.load(label)
        landmarks = landmarks.astype('float').reshape(-1, 2)
        
        sample = {'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
class RandomScale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __call__(self, sample):
        landmarks = sample['landmarks']
        
        # Pongo limites para evitar que los landmarks salgan del contorno
        min_x = np.min(landmarks[:,0]) 
        max_x = np.max(landmarks[:,0])
        ancho = max_x - min_x
        
        min_y = np.min(landmarks[:,1])
        max_y = np.max(landmarks[:,1])
        alto = max_y - min_y
                
        max_var_x = 1024 / ancho 
        max_var_y = 1024 / alto
                
        min_var_x = 0.70
        min_var_y = 0.70
                                
        varx = np.random.uniform(min_var_x, max_var_x)
        vary = np.random.uniform(min_var_x, max_var_y)
                
        landmarks[:,0] = landmarks[:,0] * varx
        landmarks[:,1] = landmarks[:,1] * vary
                
        return {'landmarks': landmarks}

                                        
class RandomMove(object):
    """Shifts segmentations a little from the center
    """
    def __call__(self, sample):
        landmarks = sample['landmarks']
        
        # To keep them in the (0, 1024) range
        
        nx, ny = np.min(landmarks, axis = 0)
        
        if nx < 0:
            landmarks[:, 0] = landmarks[:, 0] - nx
            nx = 0
        if ny < 0:
            landmarks[:, 1] = landmarks[:, 1] - ny
            ny = 0
            
        mx, my = np.max(landmarks, axis = 0)
        
        if mx > 1024:
            landmarks[:, 0] = 1024 * (landmarks[:, 0] / mx)
            mx = 1024
        if my > 1024:
            landmarks[:, 1] = 1024 * (landmarks[:, 1] / my)
            my = 1024
                    
        move_x = np.random.uniform(-nx/3, (1024-mx)/3)
        move_y = np.random.uniform(-ny/3, (1024-my)/3)
        
        landmarks[:, 0] = landmarks[:, 0] + move_x
        landmarks[:, 1] = landmarks[:, 1] + move_y
                
        return {'landmarks': landmarks}
    

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        landmarks = sample['landmarks']
        
        angle = np.random.uniform(- self.angle, self.angle)
                                        
        centro = [512, 512]
        
        landmarks -= centro
        
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        landmarks = np.dot(landmarks, R)
        
        landmarks += centro
        
        return {'landmarks': landmarks}

import cv2

def reverseVector(vector):
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    RCLAV = 23
    LCLAV = 23
    
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

    raw = np.zeros([1024, 1024])
    
    raw = drawBinary(raw, leftlung, 1)
    raw = drawBinary(raw, rightlung, 1)
    
    raw = drawBinary(raw, heart, 2)
    
    raw = drawBinary(raw, rc, 3)
    raw = drawBinary(raw, lc, 3)
    
    return raw

def getLungs(landmarks):
    leftlung, rightlung, heart, rc, lc = reverseVector(landmarks.reshape(-1))

    raw = np.zeros([1024, 1024])
    raw = drawBinary(raw, leftlung, 1)
    raw = drawBinary(raw, rightlung, 1)
    
    return raw

def getHeart(landmarks):
    leftlung, rightlung, heart, rc, lc = reverseVector(landmarks.reshape(-1))

    raw = np.zeros([1024, 1024])
    raw = drawBinary(raw, leftlung, 1)
    raw = drawBinary(raw, rightlung, 1)
    raw = drawBinary(raw, heart, 2)
    
    return raw

class ToTensorSegLungs(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        landmarks = sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        raw = getLungs(landmarks)
        
        size = raw.shape[0]
        
        landmarks = landmarks.reshape(-1, 2) / size
        landmarks = np.clip(landmarks, 0, 1)
        landmarks = landmarks[:94]
        
        return {'landmarks': torch.from_numpy(landmarks).float(),
                'seg': torch.unsqueeze(torch.from_numpy(raw),0).float()}

class ToTensorSegHeart(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        landmarks = sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        raw = getHeart(landmarks)
        
        size = raw.shape[0]
        
        landmarks = landmarks.reshape(-1, 2) / size
        landmarks = np.clip(landmarks, 0, 1)
        landmarks = landmarks[:120]
        
        return {'landmarks': torch.from_numpy(landmarks).float(),
                'seg': torch.unsqueeze(torch.from_numpy(raw),0).float()}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        landmarks = sample['landmarks']

        landmarks = landmarks.reshape(-1, 2) / 1024
        landmarks = np.clip(landmarks, 0, 1)

        return {'landmarks': torch.from_numpy(landmarks).float()}
    
class ToTensorLungs(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        landmarks = sample['landmarks']

        landmarks = landmarks.reshape(-1, 2) / 1024
        landmarks = np.clip(landmarks, 0, 1)
        landmarks = landmarks[:94]
        
        return {'landmarks': torch.from_numpy(landmarks).float()}

class ToTensorHeart(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        landmarks = sample['landmarks']

        landmarks = landmarks.reshape(-1, 2) / 1024
        landmarks = np.clip(landmarks, 0, 1)
        landmarks = landmarks[:120]
        
        return {'landmarks': torch.from_numpy(landmarks).float()}