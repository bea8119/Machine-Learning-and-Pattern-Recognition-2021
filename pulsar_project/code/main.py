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

# for K-fold
CSF_list = [(MVG.gaussianCSF_wrapper, 'Gaussian'), 
            (MVG.naiveBayesGaussianCSF, 'Naive Bayes Gaussian'), 
            (MVG.tiedCovarianceGaussianCSF, 'Tied Covariance Gaussian'),
            ]

def main():

    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')
    
    # Pre-processing (Z-normalization)
    DTR = f.Z_normalization(DTR)
    DTE = f.Z_normalization(DTE)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    # Plot distribution of attribute values (after Z-Normalizing) for each class
    # p.plotHistogram(Z_DTR, LTR, CLASS_NAMES, ATTRIBUTE_NAMES)

    # Plot heatmap of covariance
    # p.plotHeatmap(DTR, LTR)
    # show()

    # ------- MVG classifiers ------

    priorP = u.vcol(np.array([0.5, 0.5]))
    k = 2 # Number of classes

    n = 2 # Single-Fold value
    K = 3 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)
    
    # ----------------- Using validation set (single fold or K-fold) ----------------------
    # Single Fold
    idxTrain, idxTest = u.split_db_n_to_1(DTR, n)

    print(f'Single Fold ({n}-to-1) (MVG Classifiers) on Z-Normalized dataset')
    for classifier in CSF_list:
        classifier[0](DTR, LTR, k, idxTrain, idxTest, priorP, (0.5, 1, 1), show=True)
    print('-----------------------------------------------------')

    # # K-fold
    MVG.K_fold_MVG(DTR, LTR, k, priorP, K, CSF_list, (0.5, 1, 1))
    
    # ------------------ Applying PCA ------------------

    M = 7
    for m in range(5, M + 1):
        print('-----------------------------------------------------')
        # Single Fold
        DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
        PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
        DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PCA

        print(f'Single Fold (n={n}) (MVG Classifiers) on Z-Normalized dataset with PCA m={m}')
        for classifier in CSF_list:
            classifier[0](DTR_PCA, LTR, k, idxTrain, idxTest, priorP, (0.5, 1, 1), show=True)
        print('-----------------------------------------------------')

        # K-fold
        MVG.K_fold_MVG(DTR, LTR, k, priorP, K, CSF_list, (0.5, 1, 1), m)
    


    # # ------------------ Using whole Train.txt dataset and classifying Test.txt (last thing to do) --------------
    # print('-----------------------------------------------------')
    # D_merged, L_merged, idxTrain, idxTest = u.split_db_after_merge(DTR, DTE, LTR, LTE)
    # print(f'Error rates (MVG Classifiers) on Z-Normalized dataset (whole dataset)')
    # MVG.gaussianCSF_wrapper(D_merged, L_merged, k, idxTrain, idxTest, priorP, show=True)
    # MVG.naiveBayesGaussianCSF(D_merged, L_merged, k, idxTrain, idxTest, priorP, show=True)
    # MVG.tiedCovarianceGaussianCSF(D_merged, L_merged, k, idxTrain, idxTest, priorP, show=True)
    # MVG.tiedNaiveBayesGaussianCSF(D_merged, L_merged, k, idxTrain, idxTest, priorP, show=True)
    

if __name__  == '__main__':
    main()
    
