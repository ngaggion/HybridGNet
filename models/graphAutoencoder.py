import torch
import torch.nn.functional as F
from .layers import ChebConv_Coma, Pool

class GraphAutoencoder(torch.nn.Module):

    def __init__(self, config, num_features, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(GraphAutoencoder, self).__init__()

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

        # Chebyshev convolutions (encoder)
        self.cheb = torch.nn.ModuleList([
            ChebConv_Coma(
                self.filters[i],
                self.filters[i+1],
                self.K[i]
            ) for i in range(len(self.filters)-2)
        ])

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

        self.n_features_before_z = self.downsample_matrices[-1].shape[0]*self.filters[-1]

        # Fully connected layer connecting the last pooling layer and the latent space layer.
        if self.is_variational:
            self.kld_weight = config['kld_weight']
            self.enc_lin_mu = torch.nn.Linear(self.n_features_before_z, self.z)
            self.enc_lin_var = torch.nn.Linear(self.n_features_before_z, self.z)
        else:
            self.enc_lin = torch.nn.Linear(self.n_features_before_z, self.z)

        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1]*self.upsample_matrices[-1].shape[1])

        # Why this???
        self.reset_parameters()

    def encoder(self, x):
        for i in range(self.n_layers):
            x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.downsample_matrices[i])

        x = x.reshape(x.shape[0], self.n_features_before_z)
        if self.is_variational:
            mu = self.enc_lin_mu(x)
            log_var = self.enc_lin_var(x)
            return mu, log_var
        else:
            return self.enc_lin(x)

    def decoder(self, x):
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])
        return x

    def reset_parameters(self):
        if self.is_variational:
            torch.nn.init.normal_(self.enc_lin_mu.weight, 0, 0.1)
            torch.nn.init.normal_(self.enc_lin_var.weight, 0, 0.1)
        else:
            torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)


    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.filters[0])
        if self.is_variational:
            self.mu, self.log_var = self.encoder(x)
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.encoder(x)
        x = self.decoder(z)
        return x