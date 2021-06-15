import torch
import torch.nn.functional as F
from .layers import ChebConv_Coma, Pool
import torch.nn as nn

import numpy as np

from .vaeConv import EncoderConv
    
class HybridGNet(torch.nn.Module):
    def __init__(self, config, num_features, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(HybridGNet, self).__init__()

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
        if self.is_variational:
            self.mu, self.log_var = self.encoder(x)
            if self.training:
                z = self.sampling(self.mu, self.log_var)
            else:
                z = self.mu
        else:
            z = self.encoder(x)
        x = self.decoder(z)
        return x