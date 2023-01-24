import utils as u
import numpy as np
import feature_utils as f

import MVG
import SVM
import LogReg
from DCF import DCF_unnormalized_normalized_min_binary

'''
This script assumes that a fusion of Tied Full Covariance MVG and Quadratic kernel SVM (C=0.1, priorT=0.5) scores will be used
'''
saveTrainScores = True
loadTrainScores = False

saveTrainFusion = True
loadTrainFusion = False

saveTestScores = True
loadTestScores = False

saveTestFusion = True
loadTestFusion = False

scoresPath = '../data_npy/scores_'
fusionPath = '../data_npy/fusion_'

evaluation = False
seed = 0 # To use in each K-fold function and in this file to ensure same ordering of samples

PCA_list = [None, 7]
printStatus = True
calibrate = False # Calibration should be automatic due to the Fusion method 

''' SVM parameters '''

kernel_SVM = True # False for Linear SVM, regardless of the next flag
Poly_RBF = True # True for polynomial, False for RBF kernel SVM (assuming kernel flag == True)
K_svm = 1 # Value of sqrt(psi)

# Pair of (C, priorT_balance)
SVM_params = (0.1, 0.5)

# (x1T_x2 + c)^2 Polynomial kernel params
c = 1
d = 2
gamma = 1e-1

''' MVG parameters '''

k = 2 # Number of classes
CSF_list = [
    (MVG.tiedCovarianceGaussianCSF, 'Tied Full-Cov Gaussian'),
]

def main():
    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')

    # Reduced dataset (less samples) for testing only -> Comment for full training set use
    # DTR, LTR = u.reduced_dataset(DTR, LTR, 2000, seed=0)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    K = 5 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)

    # ----------------- Using validation set (K-fold) ----------------------
    for m in PCA_list:
        pca_msg = '(no PCA)' if m is None else f'(PCA m = {m})'
        if not evaluation:
            print(f'\nFusion model using {K}-Fold cross-validation scores {pca_msg}')
            print('****************************************************')
        if not loadTrainScores:

            # Compute now
            if printStatus:
                print('Computing K-fold MVG training scores...')
            MVG_scores = MVG.K_fold_MVG(DTR, LTR, k, K, CSF_list, application_points[0], m, show=False, seed=seed, calibrate=calibrate, printStatus=printStatus, returnScores=True)
            if printStatus:
                print('Computing K-fold SVM training scores...')
            SVM_scores = SVM.K_fold_SVM(DTR, LTR, K, K_svm, *SVM_params, application_points[0], m, show=False, seed=seed, kern=kernel_SVM, Poly_RBF=Poly_RBF, c=c, d=d, gamma=gamma, printStatus=printStatus, calibrate=calibrate, returnScores=True)

            if saveTrainScores:
                np.save(scoresPath + 'MVG_K_fold_PCA_{}.npy'.format(m if m is not None else 'None'), MVG_scores)
                np.save(scoresPath + 'SVM_K_fold_PCA_{}.npy'.format(m if m is not None else 'None'), SVM_scores)
        else:
            MVG_scores = np.load(scoresPath + 'MVG_K_fold_PCA_{}.npy'.format(m if m is not None else 'None'))
            SVM_scores = np.load(scoresPath + 'SVM_K_fold_PCA_{}.npy'.format(m if m is not None else 'None'))

        # Recover the order of the new samples
        np.random.seed(seed)
        idx = np.random.permutation(DTR.shape[1])

        # DCF computation
        trueL_ordered = LTR[idx] # idx was computed randomly before

        if not loadTrainFusion:
            if printStatus:
                print('Computing training Fusion model scores...')
            fused_scores, w, b = LogReg.fusionModel(MVG_scores, SVM_scores, trueL_ordered, application_points[0][0])

            if saveTrainFusion:
                np.save(scoresPath + 'Fusion_K_fold_PCA_{}.npy'.format(m if m is not None else 'None'), fused_scores)
                np.save(fusionPath + 'w_PCA_{}.npy'.format(m if m is not None else 'None'), w)
                np.save(fusionPath + 'b_PCA_{}.npy'.format(m if m is not None else 'None'), b)
        else:
            fused_scores = np.load(scoresPath + 'Fusion_K_fold_PCA_{}.npy'.format(m if m is not None else 'None'))
            w = np.load(fusionPath + 'w_PCA_{}.npy'.format(m if m is not None else 'None'))
            b = np.load(fusionPath + 'b_PCA_{}.npy'.format(m if m is not None else 'None'))

        # At this point, we have the fused scores and a transformation pair (w, b) that can be applied on the evaluation set
        if not evaluation:
            print(f'[MVG Tied Full Cov, Quad SVM (C = {SVM_params[0]}, priorT = {SVM_params[1]})]')
            for triplet in application_points:
                if printStatus:
                    print('calculating minDCF...')
                (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(fused_scores, trueL_ordered, triplet)
                print('\tpi_eff = {}\tmin DCF: {}    act DCF: {}'.format(triplet[0], round(dcf_min, 3), round(dcf_norm, 3)))
        
        else:
            # ------- Using whole Train.txt dataset and classifying Test.txt (applying [w, b] learned before) ----------------
            D_merged, L_merged, idxTR_merged, idxTE_merged = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split
            if m is not None:
                DTR_PCA_fold = u.split_dataset(D_merged, L_merged, idxTR_merged, idxTE_merged)[0][0]
                PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m)
                D_merged_PCA = np.dot(PCA_Proj.T, D_merged)
            print(f'\nFusion model on evaluation set {pca_msg}')
            print('****************************************************')
            
            if not loadTestScores:
                if printStatus:
                    print('Computing test MVG scores...')
                MVG_test_scores = CSF_list[0][0](D_merged if m is None else D_merged_PCA, L_merged, k, idxTR_merged, idxTE_merged, application_points, show=False)
                if printStatus:
                    print('Computing test SVM scores...')
                SVM_test_scores = SVM.SVM_wrapper(
                    D_merged if m is None else D_merged_PCA, L_merged, K_svm, *SVM_params, idxTR_merged, 
                    idxTE_merged, application_points[0], c, d, gamma, single_fold=False, show=False, kern=kernel_SVM, Poly_RBF=Poly_RBF,
                    calibrate=calibrate)
                if saveTestScores:
                    np.save(scoresPath + 'MVG_Test_PCA_{}.npy'.format(m if m is not None else 'None'), MVG_test_scores)
                    np.save(scoresPath + 'SVM_Test_PCA_{}.npy'.format(m if m is not None else 'None'), SVM_test_scores)
            else:
                MVG_test_scores = np.load(scoresPath + 'MVG_Test_PCA_{}.npy'.format(m if m is not None else 'None'))
                SVM_test_scores = np.load(scoresPath + 'SVM_Test_PCA_{}.npy'.format(m if m is not None else 'None'))

            if not loadTestFusion:
                # Pass w and b as arguments for applying transformation instead of performing the logreg
                if printStatus:
                    print('Computing test Fusion model scores...')
                fused_test_scores = LogReg.fusionModel(MVG_test_scores, SVM_test_scores, LTE, application_points[0][0], w, b)
                
                if saveTestFusion:
                    np.save(scoresPath + 'Fusion_Test_PCA_{}.npy'.format(m if m is not None else 'None'), fused_test_scores)
            else:
                fused_test_scores = np.load(scoresPath + 'Fusion_Test_PCA_{}.npy'.format(m if m is not None else 'None'))
            
            # At this point, we have the fused scores for the Full test set, so we perform the DCF evaluation
            print(f'[MVG Tied Full Cov, Quad SVM (C = {SVM_params[0]}, priorT = {SVM_params[1]})]')
            for triplet in application_points:
                if printStatus:
                    print('calculating minDCF...')
                (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(fused_test_scores, LTE, triplet)
                print('\tpi_eff = {}\tmin DCF: {}    act DCF: {}'.format(triplet[0], round(dcf_min, 3), round(dcf_norm, 3)))


if __name__ == '__main__':
    main()