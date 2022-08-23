import numpy as np
from utils import vcol

def Z_normalization(D):
    '''Returns Z_D (Z-transformed dataset) and the corresponding Mean and Standard Deviation'''
    mean_vector = vcol(np.mean(D, axis=1))
    std_vector = vcol(np.std(D, axis=1))

    Z_D = (D - mean_vector) / std_vector
    
    return Z_D

def centerDataset(D):
    mu_v = vcol(D.mean(axis=1))
    return D - mu_v

def covMatrix(D):
    N = D.shape[1]
    centeredD = centerDataset(D)
    return np.dot(centeredD, centeredD.T) / N

def PCA_givenM(D, M):
    C = covMatrix(D)

    _, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, :M]

    # # SVD approach
    # U, s, Vh = np.linalg.svd(C)
    # P = U[:, :m]
    return P