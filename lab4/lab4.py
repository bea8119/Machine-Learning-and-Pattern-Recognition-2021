import matplotlib.pyplot as plt
import numpy as np
import sys

# sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from lab3.lab3 import datasetCovarianceM 
from utility.vrow_vcol import vcol, vrow

def logpdf_GAU_ND(x, mu, C):
    # np.linalg.slogdet() is used because it avoids numerical errors because of small values of C (Sign LogDet)
    mu = vcol(mu) # make sure mu is (M, 1) in case it was passed as 1D array
    M = x.shape[0] # Number of rows of each sample
    N = x.shape[1] # Number of samples (columns)
    C_inv = np.linalg.inv(C)
    det_logC = np.linalg.slogdet(C)[1] # The second return value of this function (first one is the sign)

    Y = np.zeros(N)

    # IMPORTANTISIMO (Look below)

    for i in range(N):
        x_mu = vcol(x[:, i]) - mu # grab one sample only, MAKE IT A COLUMN VECTOR, NOT 1D
        mat_product = (0.5 * (np.dot(np.dot(x_mu.T, C_inv), x_mu)))
        Y[i] = - (M * 0.5) * np.log(2 * np.pi) - (0.5 * det_logC) - mat_product
    return Y

def loglikelihood(XND, m_ML, C_ML):
    N = XND.shape[1]
    return np.sum([logpdf_GAU_ND(vcol(XND[:, i]), m_ML, C_ML) for i in range(N)])

def main():
    # test 1
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    plt.figure()
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    plt.grid()
    plt.show()
    
    # test 2
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    pdfSol = np.load('./llGAU.npy')
    pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
    print(np.abs(pdfSol - pdfGau).max())

    # test 3
    XND = np.load('./XND.npy')
    mu = np.load('./muND.npy')
    C = np.load('./CND.npy')

    print(XND)
    pdfSol = np.load('./llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max())

    # likelihood test 1
    XND = np.load('./XND.npy')
    m_ML = vcol(XND.mean(axis=1))
    C_ML = datasetCovarianceM(XND)
    print(m_ML)
    print(C_ML)
    ll = loglikelihood(XND, m_ML, C_ML)
    print(ll)

    # likelihood test 2
    X1D = np.load('./X1D.npy')
    print(X1D)
    m_ML = vcol(X1D.mean(axis=1))
    C_ML = datasetCovarianceM(X1D)
    print(m_ML)
    print(C_ML)
    ll = loglikelihood(X1D, m_ML, C_ML)
    print(ll)

    # likelihood test 3
    X1D = np.load('./X1D.npy')
    m_ML = vcol(X1D.mean(axis=1))
    C_ML = datasetCovarianceM(X1D)
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
    plt.show()

if __name__ == "__main__":
    main()