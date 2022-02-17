import torch
import torch.nn as nn
import torch.nn.functional as F

from .chebConv import ChebConv, Pool

class GraphAutoencoder(nn.Module):
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices):
        super(GraphAutoencoder, self).__init__()
                      
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.kld_weight = 1e-5
               
        self.z = config['latents']
        self.n_nodes = config['n_nodes'] 
        self.filters = config['filters']
        self.K = config['K']
        
        # Genero la capa fully connected del encoder
        
        self.inLinear = self.filters[-1] * self.n_nodes[-1]   
        self.enc_lin_mu = torch.nn.Linear(self.inLinear, self.z)
        self.enc_lin_var = torch.nn.Linear(self.inLinear, self.z)
            
        # Genero la capa fully connected del decoder
     
        self.dec_lin = torch.nn.Linear(self.z, self.inLinear)
        self.reset_parameters()
                
        self.normalization1 = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization2 = torch.nn.InstanceNorm1d(self.filters[2])
        self.normalization3 = torch.nn.InstanceNorm1d(self.filters[3])
        self.normalization4 = torch.nn.InstanceNorm1d(self.filters[4])
        self.normalization5 = torch.nn.InstanceNorm1d(self.filters[5])
        self.normalization6 = torch.nn.InstanceNorm1d(self.filters[6])
        
        self.normalization2u = torch.nn.InstanceNorm1d(self.filters[1])
        self.normalization3u = torch.nn.InstanceNorm1d(self.filters[2])
        self.normalization4u = torch.nn.InstanceNorm1d(self.filters[3])
        self.normalization5u = torch.nn.InstanceNorm1d(self.filters[4])
        self.normalization6u = torch.nn.InstanceNorm1d(self.filters[5])
        
        # Guardo las capas de convoluciones en grafo
        self.graphConv_down1 = ChebConv(self.filters[0], self.filters[1], self.K)
        self.graphConv_down2 = ChebConv(self.filters[1], self.filters[2], self.K)       
        self.graphConv_down3 = ChebConv(self.filters[2], self.filters[3], self.K)
        self.graphConv_down4 = ChebConv(self.filters[3], self.filters[4], self.K)
        self.graphConv_down5 = ChebConv(self.filters[4], self.filters[5], self.K)
        self.graphConv_down6 = ChebConv(self.filters[5], self.filters[6], self.K)
        
        # Guardo las capas de convoluciones en grafo
        self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
        self.graphConv_up5 = ChebConv(self.filters[5], self.filters[4], self.K)       
        self.graphConv_up4 = ChebConv(self.filters[4], self.filters[3], self.K)
        self.graphConv_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        self.graphConv_up2 = ChebConv(self.filters[2], self.filters[1], self.K)
        
        ## Out layer: Sin bias, normalization ni relu
        self.graphConv_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias = False)
        
        self.pool = Pool()

        
    def reset_parameters(self):
        nn.init.normal_(self.enc_lin_mu.weight, 0, 0.1)
        nn.init.normal_(self.enc_lin_var.weight, 0, 0.1)
        nn.init.normal_(self.dec_lin.weight, 0, 0.1)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)       
    
    def forward(self, x):
        x = self.graphConv_down1(x, self.adjacency_matrices[0]._indices())
        x = self.normalization1(x)
        x = F.relu(x)
        
        x = self.graphConv_down2(x, self.adjacency_matrices[1]._indices())
        x = self.normalization2(x)
        x = F.relu(x)
        
        x = self.graphConv_down3(x, self.adjacency_matrices[2]._indices())
        x = self.normalization3(x)
        x = F.relu(x)
        
        x = self.pool(x, self.downsample_matrices[0])
                
        x = self.graphConv_down4(x, self.adjacency_matrices[3]._indices())
        x = self.normalization4(x)
        x = F.relu(x)
        
        x = self.graphConv_down5(x, self.adjacency_matrices[4]._indices())
        x = self.normalization5(x)
        x = F.relu(x)
                
        x = self.graphConv_down6(x, self.adjacency_matrices[5]._indices())
        x = self.normalization6(x)
        x = F.relu(x)
        
        x = x.reshape(x.shape[0], self.inLinear)
        
        self.mu = self.enc_lin_mu(x)
        self.log_var = self.enc_lin_var(x)
            
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