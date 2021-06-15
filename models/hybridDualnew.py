import torch
import torch.nn.functional as F
from .layers import ChebConv_Coma, Pool
import torch.nn as nn

import numpy as np

from .vaeConv import residualBlock

class EncoderConv(nn.Module):
    def __init__(self, latents = 128, c = 4):
        super(EncoderConv, self).__init__()
        
        self.latents = latents
        self.c = c
       
        size = self.c * np.array([2,4,8,16], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = residualBlock(1, size[0])
        self.dconv_down2 = residualBlock(size[0], size[1])
        self.dconv_down3 = residualBlock(size[1], size[2])
        self.dconv_down4 = residualBlock(size[2], size[3])
        self.dconv_down5 = residualBlock(size[3], size[3])
        
        self.fc_mu = nn.Linear(in_features=size[3]*32*32, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=size[3]*32*32, out_features=self.latents)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        x = self.dconv_down5(x)
        
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar, conv1, conv2, conv3, conv4


class DecoderConv(nn.Module):
    def __init__(self, latents = 128, c = 4, skip = False):
        super(DecoderConv, self).__init__()
        
        self.latents = latents
        self.c = c
        self.skip = skip
            
        size = self.c * np.array([2,4,8,16], dtype = np.intc)
        
        self.fc = nn.Linear(in_features=self.latents, out_features=(self.c*16)*32*32)
          
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.conv_up4 = nn.Conv2d(size[3], size[3], 3, padding=1)
        self.dconv_up4 = residualBlock(size[3], size[3])
        
        self.conv_up3 = nn.Conv2d(size[3], size[2], 3, padding=1)
        self.dconv_up3 = residualBlock(size[2], size[2])
        
        self.conv_up2 = nn.Conv2d(size[2], size[1], 3, padding=1)
        self.dconv_up2 = residualBlock(size[1], size[1])
        
        self.conv_up1 = nn.Conv2d(size[1], size[0], 3, padding=1)
        self.dconv_up1 = residualBlock(size[0], size[0])
        
        self.conv_last = nn.Conv2d(size[0], 4, 1)
        
        
    def forward(self, x, conv1, conv2, conv3, conv4):
        
        x = self.fc(x)
        x = x.view(x.size(0), self.c*16, 32, 32) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        
        x = self.upsample(x)
        x = self.conv_up4(x)
        if self.skip:
            x = torch.add(x, conv4)
        x = self.dconv_up4(x)
        
        x = self.upsample(x) 
        x = self.conv_up3(x)
        if self.skip:
            x = torch.add(x, conv3)
        x = self.dconv_up3(x)
      
        x = self.upsample(x)
        x = self.conv_up2(x)
        if self.skip:
            x = torch.add(x, conv2)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = self.conv_up1(x)
        if self.skip:
            x = torch.add(x, conv1)
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class HybridDual(torch.nn.Module):
    def __init__(self, config, num_features, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes, skip):
        super(HybridDual, self).__init__()

        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters']

        self.filters.insert(0, num_features)

        self.K = config['polygon_order']
        self.z = config['z']
        self.is_variational = (config['kld_weight'] != 0)
        if self.is_variational:
            self.kld_weight = config['kld_weight']

        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.skip = skip

        self.A_edge_index, self.A_norm = zip(
            *[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(), num_nodes[i]) for i in range(len(num_nodes))]
        )

        # Chebyshev deconvolutions (decoder)
        self.cheb_dec = torch.nn.ModuleList([
            ChebConv_Coma(
                self.filters[-i-1],
                self.filters[-i-2],
                self.K[i]
            ) for i in range(len(self.filters)-1)
        ])

        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool()

        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1]*self.upsample_matrices[-1].shape[1])

        self.encoder = EncoderConv(latents = config['z'], c = 4)
        self.decoderConv = DecoderConv(latents = config['z'], c = 4, skip = self.skip)
        
        # Why this???
        self.reset_parameters()

    def decoder(self, x):
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 


    def forward(self, x):
        self.mu, self.log_var, c1, c2, c3, c4 = self.encoder(x)
        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu

        x = self.decoder(z)
        y = self.decoderConv(z, c1, c2, c3, c4)
        
        return x, y