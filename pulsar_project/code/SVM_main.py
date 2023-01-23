import utils as u
import numpy as np
import feature_utils as f
import SVM

PCA_list = [None]

kernel_SVM = True # False for Linear SVM, regardless of the next flag
Poly_RBF = True # True for polynomial, False for RBF kernel SVM (assuming kernel flag == True)
K_svm = 1 # Value of sqrt(psi)

# Pair of (C, priorT_balance)
SVM_param_list = [
    (0.1, 0.1),
]

calibrate = True
saveScores = True

# (x1T_x2 + c)^2 Polynomial kernel params
c = 1
d = 2

# RBF parameter
gamma = 1e-1

printStatus = True

def main():

    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')

    # Reduced dataset (less samples) for testing only
    # DTR, LTR = u.reduced_dataset(DTR, LTR, 400, seed=0)

    application_points = [(0.5, 1, 1)]#, (0.1, 1, 1), (0.9, 1, 1)]

    # -------------------- Support Vector Machines ---------------------

    n = 4 # Single-Fold value
    K = 5 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)

    cal_msg = '(calibrated)' if calibrate else '(uncalibrated)'
    idxTrain_s, idxTest_s = u.split_db_n_to_1(DTR, n) # Single-fold split
    D_merged, L_merged, idxTR_merged, idxTE_merged = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split

    if kernel_SVM:
        type_SVM = '{} Kernel'.format('Quadratic' if Poly_RBF else 'RBF')
    else:
        type_SVM = 'Linear'

    for triplet in application_points:
        print('\nApplication point (pi_eff: {}, C_fn: {}, C_fp: {})'.format(*triplet))
        print('****************************************************')

        for m in PCA_list:
            pca_msg = '(no PCA)' if m is None else f'(PCA m = {m})'
            # ----------------- Using validation set (single fold or K-fold) ----------------------
            # Single Fold
            # if m is not None:
            #     DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain_s, idxTest_s)[0][0] # Retrieve single fold train subset
            #     PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
            #     DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PCA

            # print('Single Fold ({}-to-1) {} SVM {} {}'.format(
            #     n, type_SVM, pca_msg, cal_msg
            #     ))
            # for params in SVM_param_list:
            #     SVM.SVM_wrapper(DTR if m is None else DTR_PCA, LTR, K_svm, *params, idxTrain_s, idxTest_s, triplet, c, d, gamma,
            #         kern=kernel_SVM, Poly_RBF=Poly_RBF, calibrate=calibrate)
            # print('-----------------------------------------------------')

            # K-fold
            print('{}-Fold cross-validation {} SVM {} {}'.format(K, type_SVM, pca_msg, cal_msg))
            for params in SVM_param_list:
                scores = SVM.K_fold_SVM(DTR, LTR, K, K_svm, *params, triplet, m, kern=kernel_SVM, Poly_RBF=Poly_RBF, c=c, d=d, gamma=gamma, printStatus=printStatus, calibrate=calibrate, returnScores=True if saveScores else False)
                if saveScores:
                    np.save('../data_npy/scores_SVM_K_fold_PCA_{}_calibrated.npy'.format(m if m is not None else 'None'), scores)
            print('-----------------------------------------------------')

            # # # ------------------ Using whole Train.txt dataset and classifying Test.txt (last thing to do) ----------------
            # if m is not None:
            #     DTR_PCA_fold = u.split_dataset(D_merged, L_merged, idxTR_merged, idxTE_merged)[0][0]
            #     PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over training subset
            #     D_merged_PCA = np.dot(PCA_Proj.T, D_merged) # Project both training and validation subsets with the output of the PCA
            # print('{} SVM on whole dataset {}'.format(type_SVM, pca_msg))
            # for params in SVM_param_list:
            #     SVM.SVM_wrapper(
            #         D_merged if m is None else D_merged_PCA, L_merged, K_svm, *params, idxTR_merged, 
            #         idxTE_merged, triplet, c, d, gamma, single_fold=True, show=True, kern=kernel_SVM, Poly_RBF=Poly_RBF)
            # print('-----------------------------------------------------')

if __name__ == '__main__':
    main()