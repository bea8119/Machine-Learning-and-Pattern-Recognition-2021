import numpy as np
from utils import vcol

def Z_normalization(D, means=None, std_v=None):
    '''Returns Z_D (Z-transformed dataset) and the corresponding Mean and Standard Deviation vectors. 
    If applying to a evaluation/validation set, pass the vectors of the corresponding training set obtained before'''
    if means is None and std_v is None:
        mean_vector = vcol(np.mean(D, axis=1))
        std_vector = vcol(np.std(D, axis=1))
        Z_D = (D - mean_vector) / std_vector
        return Z_D, mean_vector, std_vector
    else:
        return (D - means) / std_v
    

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

def SW_compute(D, L, k):
    N = D.shape[1] # (number of total samples)

    SW = 0
    for i in range(k):
        # Use samples of each different (c)lass
        Dc = D[:, L == i]
        nc = Dc.shape[1] # n of samples per class
        # Covariance matrix of each class
        Sw = covMatrix(Dc)

        SW += (Sw * nc)

    SW /= N

    return SW