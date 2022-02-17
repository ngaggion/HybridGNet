import numpy as np
import scipy.sparse as sp
import torch

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

## Adjacency Matrix
def mOrgan(N):
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i-1] = 1
        sub[i, (i+1)%N] = 1
    return sub

## Downsampling Matrix
def mOrganD(N):
    N2 = int(np.ceil(N/2))
    sub = np.zeros([N2, N])
    
    for i in range(0, N2):
        if (2*i+1) == N:
            sub[i, 2*i] = 1
        else:
            sub[i, 2*i] = 1/2
            sub[i, 2*i+1] = 1/2
            
    return sub

## Upsampling Matrix
def mOrganU(N):
    N2 = int(np.ceil(N/2))
    sub = np.zeros([N, N2])
    
    for i in range(0, N):
        if i % 2 == 0 or (i//2)+1 == N2:
            sub[i, i//2] = 1
        else:
            sub[i, i//2] = 1/2
            sub[i, i//2 + 1] = 1/2
            
    return sub

## Generating Matrixes for every organ
def genMatrixes():       
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    CLA1 = 23
    CLA2 = 23
    
    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    Asub3 = mOrgan(HEART)
    Asub4 = mOrgan(CLA1)
    Asub5 = mOrgan(CLA2)
    
    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
    ADsub3 = mOrgan(int(np.ceil(HEART / 2)))
    ADsub4 = mOrgan(int(np.ceil(CLA1 / 2)))
    ADsub5 = mOrgan(int(np.ceil(CLA2 / 2)))
                    
    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    Dsub3 = mOrganD(HEART)
    Dsub4 = mOrganD(CLA1)
    Dsub5 = mOrganD(CLA2) 
    
    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
    Usub3 = mOrganU(HEART)
    Usub4 = mOrganU(CLA1)
    Usub5 = mOrganU(CLA2)    
        
    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART
    p4 = p3 + CLA1
    p5 = p4 + CLA2
    
    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    p3_ = p2_ + int(np.ceil(HEART / 2))
    p4_ = p3_ + int(np.ceil(CLA1 / 2))
    p5_ = p4_ + int(np.ceil(CLA2 / 2))
    
    A = np.zeros([p5, p5])
    
    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    A[p2:p3, p2:p3] = Asub3
    A[p3:p4, p3:p4] = Asub4
    A[p4:p5, p4:p5] = Asub5
    
    AD = np.zeros([p5_, p5_])
    
    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
    AD[p2_:p3_, p2_:p3_] = ADsub3
    AD[p3_:p4_, p3_:p4_] = ADsub4
    AD[p4_:p5_, p4_:p5_] = ADsub5
   
    D = np.zeros([p5_, p5])
    
    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    D[p2_:p3_, p2:p3] = Dsub3
    D[p3_:p4_, p3:p4] = Dsub4
    D[p4_:p5_, p4:p5] = Dsub5
    
    U = np.zeros([p5, p5_])
    
    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    U[p2:p3, p2_:p3_] = Usub3
    U[p3:p4, p3_:p4_] = Usub4
    U[p4:p5, p4_:p5_] = Usub5
    
    return A, AD, D, U


def genMatrixesLungs():       
    RLUNG = 44
    LLUNG = 50
    
    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    
    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
                    
    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    
    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
        
    p1 = RLUNG
    p2 = p1 + LLUNG
    
    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    
    A = np.zeros([p2, p2])
    
    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    
    AD = np.zeros([p2_, p2_])
    
    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
   
    D = np.zeros([p2_, p2])
    
    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    
    U = np.zeros([p2, p2_])
    
    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    
    return A, AD, D, U


def genMatrixesLH():       
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    
    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    Asub3 = mOrgan(HEART)
    
    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
    ADsub3 = mOrgan(int(np.ceil(HEART / 2)))
                    
    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    Dsub3 = mOrganD(HEART)
    
    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
    Usub3 = mOrganU(HEART)
        
    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART
    
    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    p3_ = p2_ + int(np.ceil(HEART / 2))
    
    A = np.zeros([p3, p3])
    
    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    A[p2:p3, p2:p3] = Asub3
    
    AD = np.zeros([p3_, p3_])
    
    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
    AD[p2_:p3_, p2_:p3_] = ADsub3
   
    D = np.zeros([p3_, p3])
    
    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    D[p2_:p3_, p2:p3] = Dsub3
    
    U = np.zeros([p3, p3_])
    
    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    U[p2:p3, p2_:p3_] = Usub3
    
    return A, AD, D, U
