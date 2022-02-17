import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .vaeConv import residualBlock

class EncoderConv(nn.Module):
    def __init__(self, latents = 64, hw = 16):
        super(EncoderConv, self).__init__()
        
        self.latents = latents
        self.c = 4
        
        size = self.c * np.array([2,4,8,16,32], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = residualBlock(1, size[0])
        self.dconv_down2 = residualBlock(size[0], size[1])
        self.dconv_down3 = residualBlock(size[1], size[2])
        self.dconv_down4 = residualBlock(size[2], size[3])
        self.dconv_down5 = residualBlock(size[3], size[4])
        self.dconv_down6 = residualBlock(size[4], size[4])
        
        self.fc_mu = nn.Linear(in_features=size[4]*hw*hw, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=size[4]*hw*hw, out_features=self.latents)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        x = self.dconv_down2(x)
        x = self.maxpool(x)
        
        x = self.dconv_down3(x)
        x = self.maxpool(x)
        
        x = self.dconv_down4(x)
        x = self.maxpool(x)
        
        x = self.dconv_down5(x)
        x = self.maxpool(x)
        
        x = self.dconv_down6(x)
        
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        x_mu = self.fc_mu(x)
        
        return x_mu

    
class DecoderPCA(nn.Module):
    def __init__(self, config, latents=64):
        super(DecoderPCA, self).__init__()
        
        device = config['device']
        
        if config['extended']:
            self.matrix = np.load('models/pca_components_ext.npy')
            self.mean = np.load('models/pca_mean_ext.npy')
        else:
            self.matrix = np.load('models/pca_components_JSRT.npy')
            self.mean = np.load('models/pca_mean_JSRT.npy')

        self.matrix = torch.from_numpy(self.matrix).float().to(device)
        self.mean = torch.from_numpy(self.mean).float().to(device)
        
    def forward(self, x):
        x = torch.matmul(x, self.matrix) + self.mean
        
        return x

class PCA_Net(nn.Module):
    def __init__(self, config):
        super(PCA_Net, self).__init__()
                       
        hw = config['inputsize'] // 32
        self.z = config['latents']
        
        self.encoder = EncoderConv(latents = self.z, hw = hw)
        self.decoder = DecoderPCA(config)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
