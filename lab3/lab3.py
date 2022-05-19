import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import sys
sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR/')
from lab2.load_plot import load
from utility.vrow_vcol import vcol

def datasetCovarianceM(dataset):
    # dataset mean computation
    mu = dataset.mean(axis=1) # compute mean of columns -> 1D array returned

    # centered dataset
    dataCentered = dataset - vcol(mu)
    N = dataCentered.shape[1] # shape is an attribute, remember
    return np.dot(dataCentered, dataCentered.T) / N

    
def PCA_PgivenM(dataset, m):
    
    C = datasetCovarianceM(dataset)

    # eigh assumes the matrix is symmetric and returns SORTED eigenvalues
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, :m] # reverse the columns first, then grab M columns

    # also you can do it like this (with SVD approach)
    # U, s, Vh = np.linalg.svd(C)
    # P = U[:, :m]

    Dprojected = np.dot(P.T, dataset)
    return Dprojected

def plotIRISPCA(D, L):
    DPCA = PCA_PgivenM(D, 2)
    # we want to plot in 2D, even though we would have 4 separate dimensions to plot
    plt.figure()

    plt.scatter(DPCA[0, L == 0], DPCA[1, L == 0], label='Setosa')
    plt.scatter(DPCA[0, L == 1], DPCA[1, L == 1], label='Versicolor')
    plt.scatter(DPCA[0, L == 2], DPCA[1, L == 2], label='Virginica')
    plt.legend()
    plt.grid()
    plt.show()

def SBandSW(dataset, labels, K): # K is the number of classes
    N = dataset.shape[1] # (number of total samples)
    mu = vcol(dataset.mean(axis=1)) # mean of the whole dataset (casted to column vector)

    SW = 0
    for i in range(K):
        # Use samples of each different (c)lass
        Dc = dataset[:, labels == i]
        nc = Dc.shape[1] # n of samples per class
        # Covariance matrix of each class
        Sw = datasetCovarianceM(Dc)

        SW += (Sw * nc)

    SW /= N 

    SB = 0
    for i in range(K):
        Dc = dataset[:, labels == i]
        nc = Dc.shape[1]
        mu_c = vcol(Dc.mean(axis=1)) # don't forget to reshape as column vector
        
        SB += nc * (np.dot((mu_c - mu), (mu_c - mu).T))
    
    SB /= N

    return SB, SW

# def LDA(dataset, labels, m):
def main():
    D, L = load('iris.csv')
    SB, SW = SBandSW(D, L, 3)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[: ,::-1][:, 0:m]
    UW, _, _ = np.linalg.svd(W) # Underscore means you don't care about this
    U = UW[:, 0:m]
    
    # Generalized eigenvalue problem (joint diagonalization) 
    
main()
