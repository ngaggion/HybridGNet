import numpy as np
import scipy.sparse as sp
import torch

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def normal(tensor, mean, std):
    if tensor is not None:
        torch.nn.init.normal_(tensor, mean=mean, std=std)

        
def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def mOrgan(N):
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i-1] = 1
        sub[i, (i+1)%N] = 1
    return sub

def genAdyacencyMatrix():
    A = np.zeros([166, 166])
        
    RLUNG = 44
    sub1 = mOrgan(RLUNG)
    LLUNG = 50
    sub2 = mOrgan(LLUNG)
    HEART = 26
    sub3 = mOrgan(HEART)
    CLA1 = 23
    sub4 = mOrgan(CLA1)
    CLA2 = 23
    sub5 = mOrgan(CLA2)
    
    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART
    p4 = p3 + CLA1
    p5 = p4 + CLA2
    
    A[:p1, :p1] = sub1
    A[p1:p2, p1:p2] = sub2
    A[p2:p3, p2:p3] = sub3
    A[p3:p4, p3:p4] = sub4
    A[p4:p5, p4:p5] = sub5
   
    return A
