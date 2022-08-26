import utils as u
import feature_utils as f
import LogReg as LR
import numpy as np

LR_param_list = [
    (1e-5, 0.5),
    (1e-5, 0.1),
    (1e-5, 0.9),
] # Pairs of parameters to apply LogReg

PCA_list = [None, 7, 6, 5]

def main():
    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')
    
    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    # ----------------- Logistic Regression -------------------

    n = 4 # Single-Fold value
    K = 5 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)

    idxTrain_s, idxTest_s = u.split_db_n_to_1(DTR, n) # Single-fold split
    D_merged, L_merged, idxTR_merged, idxTE_merged = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split

    for triplet in application_points:
        # ----------------- Using validation set (single fold or K-fold) ----------------------
        print('\nApplication point (pi_eff: {}, C_fn: {}, C_fp: {})'.format(*triplet))
        print('****************************************************')

        for m in PCA_list:
            # Single Fold
            if m is not None:
                DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain_s, idxTest_s)[0][0] # Retrieve single fold train subset
                PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
                DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PCA

            print('Single Fold ({}-to-1) Linear Log Reg {}'.format(n, '(no PCA)' if m is None else f'(PCA m = {m})'))
            for params in LR_param_list:
                LR.logReg_wrapper(DTR if m is None else DTR_PCA, LTR, *params, idxTrain_s, idxTest_s, triplet)
            print('-----------------------------------------------------')

            # # K-fold
            # LR.K_fold_LogReg(DTR, LTR, K, LR_param_list, triplet, m)
            # print('-----------------------------------------------------')

            # ------------------ Using whole Train.txt dataset and classifying Test.txt (last thing to do) ----------------
            if m is not None:
                DTR_PCA_fold = u.split_dataset(D_merged, L_merged, idxTR_merged, idxTE_merged)[0][0]
                PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over training subset
                D_merged_PCA = np.dot(PCA_Proj.T, D_merged) # Project both training and validation subsets with the output of the PCA
            print('Log Reg on whole dataset {}'.format('(no PCA)' if m is None else f'(PCA m = {m})'))
            for params in LR_param_list:
                LR.logReg_wrapper(D_merged if m is None else D_merged_PCA, L_merged, *params, idxTR_merged, idxTE_merged, triplet)
            print('-----------------------------------------------------')

if __name__ == '__main__':
    main()