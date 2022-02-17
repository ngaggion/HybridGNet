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
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar

    
    
class DecoderFull(nn.Module):
    def __init__(self, config, neurons = 512):
        super(DecoderFull, self).__init__()
        self.latents = config['latents']
        self.neurons = neurons
        
        self.nonLinearity = nn.ReLU()
        self.fc1 = nn.Linear(self.latents, self.neurons)
        
        if config['allOrgans'] == False:
            output_size = 240
        else:
            output_size = 332
            
        self.fcout = nn.Linear(in_features=self.neurons, out_features=output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nonLinearity(x)
        
        out = self.fcout(x)
        
        return out

            
class VAE_Mixed(nn.Module):
    def __init__(self, config):
        super(VAE_Mixed, self).__init__()
        latents = config['latents']
        
        hw = config['inputsize'] // 32
        
        self.encoder = EncoderConv(latents, hw)
        self.decoder = DecoderFull(config, neurons = 512)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)   
    
    def forward(self, x):
        self.mu, self.log_var = self.encoder(x)

        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu
            
        x_recon = self.decoder(z)
        return x_recon