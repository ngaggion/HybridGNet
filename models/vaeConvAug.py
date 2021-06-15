import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(residualBlock, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False))
        else:
          self.skip = None

        self.block = nn.Sequential(nn.BatchNorm2d(in_channels, track_running_stats=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.BatchNorm2d(out_channels, track_running_stats=False),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 3, padding=1)
                                   )   

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = F.relu(out)

        return out

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
        
        x = self.dconv_down4(x)
        x = self.maxpool(x)
        
        x = self.dconv_down5(x)
        
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        
        return x_mu, x_logvar

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
        self.conv_last = nn.Conv2d(size[0], 1, 1)
        
        
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

class VariationalAutoencoderConv(nn.Module):
    def __init__(self, latents = 128, c = 4):
        super(VariationalAutoencoderConv, self).__init__()
        self.encoder = EncoderConv(latents, c)
        self.decoder = DecoderConv(latents, c)
    
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
        
def vae_loss(recon_x, x, mu, logvar, variational_beta):
    loss_rec = F.mse_loss(recon_x, x) 
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_rec + variational_beta * kldivergence
    
    return loss