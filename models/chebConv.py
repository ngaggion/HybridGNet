from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.nn.inits import zeros, normal

# We change the default initialization from zeros to a normal distribution
class ChebConv(ChebConv):
    def reset_parameters(self):
        for lin in self.lins:
            normal(lin, mean = 0, std = 0.1)
            #lin.reset_parameters()
        normal(self.bias, mean = 0, std = 0.1)
        #zeros(self.bias)

# Pooling from COMA: https://github.com/pixelite1201/pytorch_coma/blob/master/layers.py
class Pool(MessagePassing):
    def __init__(self):
        # source_to_target is the default value for flow, but is specified here for explicitness
        super(Pool, self).__init__(flow='source_to_target')

    def forward(self, x, pool_mat,  dtype=None):
        pool_mat = pool_mat.transpose(0, 1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out

    def message(self, x_j, norm):
        return norm.view(1, -1, 1) * x_j