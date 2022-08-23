import utils as u
import feature_utils as f
import plotting as p
import MVG
from matplotlib.pyplot import show
import numpy as np

CLASS_NAMES = ['RFI / Noise', 'Pulsar']
ATTRIBUTE_NAMES = ['Mean of the integrated profile',
                 'Standard deviation of the integrated profile',
                 'Excess kurtosis of the integrated profile',
                 'Skewness of the integrated profile',
                 'Mean of the DM-SNR curve',
                 'Standard deviation of the DM-SNR curve',
                 'Excess kurtosis of the DM-SNR curve',
                 'Skewness of the DM-SNR curve']

def main():

    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')

    Z_DTR = f.Z_normalization(DTR)

    # Plot distribution of attribute values (after Z-Normalizing) for each class
    # p.plotHistogram(Z_DTR, LTR, CLASS_NAMES, ATTRIBUTE_NAMES)

    # Plot heatmap of covariance
    # p.plotHeatmap(DTR, LTR)
    # show()

    # Apply PCA
    M = 6
    PCA_ProjM = f.PCA_givenM(Z_DTR, M)

    DTR_PCA = np.dot(PCA_ProjM.T, DTR)
    Z_DTR_PCA = np.dot(PCA_ProjM.T, Z_DTR)

    # ------- MVG classifiers ------

    priorP = u.vcol(np.array([0.5, 0.5]))
    k = 2 # Number of classes

    # Single Fold
    n = 4 # Single-Fold value
    idxTrain, idxTest = u.split_db_n_to_1(DTR, n)
    print(f'Single Fold (n={n}) error rates (MVG Classifiers)')
    MVG.gaussianCSF_wrapper(DTR, LTR, k, idxTrain, idxTest, priorP, show=True)
    MVG.naiveBayesGaussianCSF(DTR, LTR, k, idxTrain, idxTest, priorP, show=True)
    MVG.tiedCovarianceGaussianCSF(DTR, LTR, k, idxTrain, idxTest, priorP, show=True)
    MVG.tiedNaiveBayesGaussianCSF(DTR, LTR, k, idxTrain, idxTest, priorP, show=True)
    print('--------------------------------------')

    # K-fold
    CSF_list = [(MVG.gaussianCSF_wrapper, 'Gaussian'), 
                (MVG.naiveBayesGaussianCSF, 'Naive Bayes Gaussian'), 
                (MVG.tiedCovarianceGaussianCSF, 'Tied Covariance Gaussian'), 
                (MVG.tiedNaiveBayesGaussianCSF, 'Tied Naive Bayes Gaussian')]

    K = 5 # Leave-One-Out if equal to D.shape[1] (number of samples)
    MVG.K_fold_MVG(DTR, LTR, k, priorP, K, CSF_list)

if __name__  == '__main__':
    main()
    
