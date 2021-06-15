#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:06:46 2021

@author: ngaggion
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .vaeConv import EncoderConv 


class EncoderFull(nn.Module):
    def __init__(self, input_size = 240, latents = 128, neurons = 256):
        self.latents = latents
        self.neurons = neurons
        super(EncoderFull, self).__init__()
        
        self.nonLinearity = nn.ReLU()
        self.fc1 = nn.Linear(input_size, self.neurons)
        #self.drop1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(self.neurons, self.neurons)
        #self.drop2 = nn.Dropout(0.20) 
        
        self.fc_mu = nn.Linear(in_features=self.neurons, out_features=self.latents)
        self.fc_logvar = nn.Linear(in_features=self.neurons, out_features=self.latents)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nonLinearity(x)
        #x = self.drop1(x)

        x = self.fc2(x)
        x = self.nonLinearity(x)
        #x = self.drop2(x)
              
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar
    
class DecoderFull(nn.Module):
    def __init__(self, input_size = 240, latents = 128, neurons = 256):
        super(DecoderFull, self).__init__()
        self.latents = latents
        self.neurons = neurons
        
        self.nonLinearity = nn.ReLU()
        self.fc1 = nn.Linear(self.latents, self.neurons)
        self.fc2 = nn.Linear(self.neurons, self.neurons)
        
        self.fcout = nn.Linear(in_features=self.neurons, out_features=input_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nonLinearity(x)

        x = self.fc2(x)
        x = self.nonLinearity(x)
        
        out = self.fcout(x)
        
        return out

    
class VariationalAutoencoderFC(nn.Module):
    def __init__(self, inputsize = 240, latents = 128, neurons = 256):
        super(VariationalAutoencoderFC, self).__init__()
        self.encoder = EncoderFull(inputsize, latents, neurons)
        self.decoder = DecoderFull(inputsize, latents, neurons)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
                
            
class VariationalAutoencoderMixto(nn.Module):
    def __init__(self, inputsize = 240, latents = 128, c = 4):
        super(VariationalAutoencoderMixto, self).__init__()
        self.encoderconv = EncoderConv(latents, c)
        self.decoder = DecoderFull(inputsize, latents, neurons = 256)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoderconv(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

        

def vae_loss(recon_x, x, mu, logvar, variational_beta):
    loss_rec = F.mse_loss(recon_x, x) 
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_rec + variational_beta * kldivergence
    
    return loss
