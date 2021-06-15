import torch
import torch.nn.functional as F
from .layers import ChebConv_Coma, Pool
import torch.nn as nn

import numpy as np

from .vaeConv import EncoderConv, residualBlock

class DecoderConv(nn.Module):
    def __init__(self, latents = 128, c = 4):
        super(DecoderConv, self).__init__()
        
        self.latents = latents
        self.c = c
        
        size = self.c * np.array([2,4,8,16], dtype = np.intc)
        
        self.fc = nn.Linear(in_features=self.latents, out_features=(self.c*16)*32*32)
          
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up4 = residualBlock(size[3], size[3])
        self.dconv_up3 = residualBlock(size[3], size[2])
        self.dconv_up2 = residualBlock(size[2], size[1])
        self.dconv_up1 = residualBlock(size[1], size[0])
        self.conv_last = nn.Conv2d(size[0], 4, 1)
        
    def forward(self, x):        
        x = self.fc(x)
        x = x.view(x.size(0), self.c*16, 32, 32) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        
        x = self.upsample(x)
        x = self.dconv_up4(x)
        
        x = self.upsample(x)    
        x = self.dconv_up3(x)
      
        x = self.upsample(x)   
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out

class AutoencoderSeg(torch.nn.Module):

    def __init__(self):
        super(AutoencoderSeg, self).__init__()

        self.encoder = EncoderConv(latents = 64, c = 4)
        self.decoderConv = DecoderConv(latents = 64, c = 4)
        self.is_variational = True

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 

    def forward(self, x):
        if self.is_variational:
            self.mu, self.log_var = self.encoder(x)
            if self.training:
                z = self.sampling(self.mu, self.log_var)
            else:
                z = self.mu
        else:
            z = self.encoder(x)

        y = self.decoderConv(z)
        
        return y