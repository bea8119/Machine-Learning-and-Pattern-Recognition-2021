import utils as u
import feature_utils as f
import plotting as p
import MVG
import numpy as np

# for K-fold
CSF_list = [
            (MVG.gaussianCSF_wrapper, 'Full Covariance Gaussian'), 
            (MVG.naiveBayesGaussianCSF, 'Diag Covariance Gaussian'), 
            (MVG.tiedCovarianceGaussianCSF, 'Tied Covariance Gaussian'),
            ]

def main():

    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')
    
    # Pre-processing (Z-normalization)
    DTR = f.Z_normalization(DTR)
    DTE = f.Z_normalization(DTE)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    # ------- MVG classifiers ------

    priorP = u.vcol(np.array([0.5, 0.5]))
    k = 2 # Number of classes

    n = 4 # Single-Fold value
    K = 5 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)
    
    for triplet in application_points:
        # ----------------- Using validation set (single fold or K-fold) ----------------------
        # Single Fold
        print('\nApplication point (pi_eff: {}, C_fn: {}, C_fp: {})'.format(*triplet))
        print('****************************************************')
        idxTrain, idxTest = u.split_db_n_to_1(DTR, n)
        print(f'Single fold ({n}-to-1) (MVG Classifiers) (No PCA)')
        for classifier in CSF_list:
            classifier[0](DTR, LTR, k, idxTrain, idxTest, priorP, triplet, show=True)
        print('-----------------------------------------------------')

        # K-fold
        MVG.K_fold_MVG(DTR, LTR, k, priorP, K, CSF_list, triplet)

        # ------------------ Applying PCA ------------------

        M = 7
        for m in range(M, 3, -1):
            print('-----------------------------------------------------')
            # Single Fold
            DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
            PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
            DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PCA

            print(f'Single Fold (n={n}) (MVG Classifiers) with PCA m={m}')
            for classifier in CSF_list:
                classifier[0](DTR_PCA, LTR, k, idxTrain, idxTest, priorP, triplet, show=True)
            print('-----------------------------------------------------')

            # K-fold
            MVG.K_fold_MVG(DTR, LTR, k, priorP, K, CSF_list, triplet, m)
        

        # ------------------ Using whole Train.txt dataset and classifying Test.txt (last thing to do) --------------
        # print('-----------------------------------------------------')
        # D_merged, L_merged, idxTrain, idxTest = u.split_db_after_merge(DTR, DTE, LTR, LTE)
        # print(f'MVG Classifiers on whole dataset)')
        # MVG.gaussianCSF_wrapper(D_merged, L_merged, k, idxTrain, idxTest, priorP, triplet, show=True)
        # MVG.naiveBayesGaussianCSF(D_merged, L_merged, k, idxTrain, idxTest, priorP, triplet, show=True)
        # MVG.tiedCovarianceGaussianCSF(D_merged, L_merged, k, idxTrain, idxTest, priorP, triplet, show=True)
    

if __name__  == '__main__':
    main()
    
