import torch
import torch.nn as nn
import torch.nn.functional as F

from .chebConv import ChebConv, Pool

from .modelutils import residualBlock

import numpy as np

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
    
    
class Hybrid(nn.Module):
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
        super(Hybrid, self).__init__()
               
        hw = config['inputsize'] // 32
        self.z = config['latents']
        self.encoder = EncoderConv(latents = self.z, hw = hw)
        
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.kld_weight = 1e-5
                
        n_nodes = config['n_nodes']
        self.filters = config['filters']
        self.K = config['K'] # orden del polinomio
        
        # Genero la capa fully connected del decoder
        outshape = self.filters[-1] * n_nodes[-1]        
        self.dec_lin = torch.nn.Linear(self.z, outshape)
                                
        self.normalization2u = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization3u = torch.nn.InstanceNorm1d(self.filters[2])
        self.normalization4u = torch.nn.InstanceNorm1d(self.filters[3])
        self.normalization5u = torch.nn.InstanceNorm1d(self.filters[4])
        self.normalization6u = torch.nn.InstanceNorm1d(self.filters[5])
        
        self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
        self.graphConv_up5 = ChebConv(self.filters[5], self.filters[4], self.K)       
        self.graphConv_up4 = ChebConv(self.filters[4], self.filters[3], self.K)
        self.graphConv_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        self.graphConv_up2 = ChebConv(self.filters[2], self.filters[1], self.K)
        
        ## Out layer: Sin bias, normalization ni relu
        self.graphConv_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias = False)
        
        self.pool = Pool()
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

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
            
        x = self.dec_lin(z)
        x = F.relu(x)
                        
        x = x.reshape(x.shape[0], -1, self.filters[5])
                
        x = self.graphConv_up6(x, self.adjacency_matrices[5]._indices())
        x = self.normalization6u(x)
        x = F.relu(x)
        
        x = self.graphConv_up5(x, self.adjacency_matrices[4]._indices())
        x = self.normalization5u(x)
        x = F.relu(x)
        
        x = self.graphConv_up4(x, self.adjacency_matrices[3]._indices())
        x = self.normalization4u(x)
        x = F.relu(x)
        
        x = self.pool(x, self.upsample_matrices[0])
        
        x = self.graphConv_up3(x, self.adjacency_matrices[2]._indices())
        x = self.normalization3u(x)
        x = F.relu(x)
        
        x = self.graphConv_up2(x, self.adjacency_matrices[1]._indices())
        x = self.normalization2u(x)
        x = F.relu(x)
        
        x = self.graphConv_up1(x, self.adjacency_matrices[0]._indices()) # Sin relu y sin bias
        
        return x