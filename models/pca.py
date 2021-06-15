import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .vaeConv import residualBlock

class EncoderConv(nn.Module):
    def __init__(self, latents = 64):
        super(EncoderConv, self).__init__()
        c = 4
        latent_dims = latents
        
        size = c * np.array([2,4,8,16], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = residualBlock(1, size[0])
        self.dconv_down2 = residualBlock(size[0], size[1])
        self.dconv_down3 = residualBlock(size[1], size[2])
        self.dconv_down4 = residualBlock(size[2], size[3])
        self.dconv_down5 = residualBlock(size[3], size[3]) 
        
        self.fc = nn.Linear(in_features=size[3]*32*32, out_features=latent_dims)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
      
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        x = self.maxpool(x)
        
        x = self.dconv_down5(x)
        
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        x = self.fc(x)
        
        return x
    
class DecoderPCA(nn.Module):
    def __init__(self, latents=64):
        super(DecoderPCA, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if latents == 64:
            self.matrix = np.load('models/pca64.npy')
            self.mean = np.load('models/mean_pca64.npy')
        else:
            self.matrix = np.load('models/pca128.npy')
            self.mean = np.load('models/mean_pca128.npy')
            
        self.matrix = torch.from_numpy(self.matrix).float().to(device)
        self.mean = torch.from_numpy(self.mean).float().to(device)
        
    def forward(self, x):
        x = torch.matmul(x, self.matrix) + self.mean
        
        return x

class PCA_Net(nn.Module):
    def __init__(self, latents = 64):
        super(PCA_Net, self).__init__()
        self.encoderconv = EncoderConv(latents)
        self.decoder = DecoderPCA(latents)
    
    def forward(self, x):
        x = self.encoderconv(x)
        x = self.decoder(x)
        return x