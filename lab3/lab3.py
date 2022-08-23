import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import sys
sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
# sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from lab2.load_plot import load
from utility.vrow_vcol import vcol, vrow

def datasetCovarianceM(dataset):
    # dataset mean computation
    mu = dataset.mean(axis=1) # compute mean of columns -> 1D array returned

    # centered dataset
    dataCentered = dataset - vcol(mu)
    N = dataCentered.shape[1] # shape is an attribute, remember
    return np.dot(dataCentered, dataCentered.T) / N


def PCA_PgivenM(dataset, m): # Returns projection matrix P

    C = datasetCovarianceM(dataset)

    # eigh assumes the matrix is symmetric and returns SORTED eigenvalues
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, :m] # reverse the columns first, then grab M columns

    # also you can do it like this (with SVD approach)
    # U, s, Vh = np.linalg.svd(C)
    # P = U[:, :m]
    return P

def plotIRIS2D(DProjected, L, name):
    # we want to plot in 2D, even though we would have 4 separate dimensions to plot
    plt.figure(name)

    plt.scatter(DProjected[0, L == 0], DProjected[1, L == 0], label='Setosa')
    plt.scatter(DProjected[0, L == 1], DProjected[1, L == 1], label='Versicolor')
    plt.scatter(DProjected[0, L == 2], DProjected[1, L == 2], label='Virginica')
    plt.legend()
    plt.grid()
    # plt.show()

def SW_compute(dataset, labels, K):
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

    return SW

def SB_compute(dataset, labels, K):
    N = dataset.shape[1]
    mu = vcol(dataset.mean(axis=1))

    SB = 0
    for i in range(K):
        Dc = dataset[:, labels == i]
        nc = Dc.shape[1]
        mu_c = vcol(Dc.mean(axis=1)) # don't forget to reshape as column vector

        SB += nc * (np.dot((mu_c - mu), (mu_c - mu).T))

    SB /= N

    return SB


def LDA_WgivenM(dataset, labels, m, K): # returns projection matrix W -> at most K-1 discriminant directions, receives K number of classes
    SB = SB_compute(dataset, labels, K)
    SW = SW_compute(dataset, labels, K)

    # Method 1: Generalized eig. problem

    # s, U = scipy.linalg.eigh(SB, SW)
    # W = U[: ,::-1][:, 0:4] 
    # 
    # # END of solution, but you can get a basis with the following 2 lines

    # UW, _, _ = np.linalg.svd(W) # Underscore means you don't care about this
    # U = UW[:, 0:m] # U is column of eigenvectors


    # Method 2: Generalized eigenvalue problem (joint diagonalization of SB and SW)
    U, s, _ = np.linalg.svd(SW)
    sigmaDiag = 1.0 / (s**0.5)
    P1 = np.dot(U * vrow(sigmaDiag), U.T) # * is an operation with broadcasting 
    # (Every element of the eigenvectors is multiplied by the corresponding diagonal entry)
    # It's a faster way of doing matrix multiplication with DIAGONAL matrix
    
    # the other way would be this:
    # P1 = np.dot(np.dot(U, np.diag(1.0 / (s**0.5))), U.T)

    # Transformed between class covariance
    SBT = np.dot(np.dot(P1, SB), P1.T)

    P2 = np.linalg.svd(SBT)[0][:, :m] # Take [0] of the 3 returns of .svd and slice to get up to m eigenvectors

    # LDA matrix
    W = np.dot(P1.T, P2)

    return W

def main():
    D, L = load('../datasets/iris.csv')
    K = len(np.unique(L)) # Number of classes
    P = PCA_PgivenM(D, 2)
    W = LDA_WgivenM(D, L, 2, K)

    # Project dataset onto the transformation matrices
    DPCA = np.dot(P.T, D)
    DLDA = np.dot(W.T, D) # Has flipped sign wrt to prof's solution (totally fine)

    solution = np.load('IRIS_LDA_matrix_m2.npy')
    DLDAsol = np.dot(solution.T, D)

    plotIRIS2D(DPCA, L, 'PCA')
    plotIRIS2D(DLDA, L, 'LDA - my solution')
    plotIRIS2D(DLDAsol, L, 'LDA - prof\'s')

    # print(np.linalg.svd(np.hstack([W, solution]))[1]) # Test to check for equality of represented subspace PASSED
    plt.show()

if __name__ == "__main__":
    main()

