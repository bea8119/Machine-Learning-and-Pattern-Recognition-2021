import utils as u
import feature_utils as f
import MVG
import LogReg as LR
import numpy as np

LR_param_list = [
    (1e-5, 0.5),
    (1e-5, 0.1),
    (1e-5, 0.9),
] # Pairs of parameters to apply LogReg

def main():
    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')
    
    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    # ----------------- Logistic Regression -------------------

    n = 4 # Single-Fold value
    K = 5 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)
    
    l = 1e-5 # Lambda hyperparameter
    priorT = 0.5

    for triplet in application_points:
        # # ----------------- Using validation set (single fold or K-fold) ----------------------
        # print('\nApplication point (pi_eff: {}, C_fn: {}, C_fp: {})'.format(*triplet))
        # print('****************************************************')

        # # Single Fold
        idxTrain, idxTest = u.split_db_n_to_1(DTR, n)
        # print(f'Single fold ({n}-to-1) Linear Log-Reg (no PCA)')
        
        # for params in LR_param_list:
        #     LR.logReg_wrapper(DTR, LTR, *params, idxTrain, idxTest, triplet)
        # print('-----------------------------------------------------')

        # # K-fold
        # LR.K_fold_LogReg(DTR, LTR, K, LR_param_list, triplet)

        # ------------------ Applying PCA ------------------

        M = 7
        for m in range(M, 3, -1):
            print('-----------------------------------------------------')
            # Single Fold
            DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
            PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
            DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PCA

            print(f'Single Fold ({n}-to-1) Linear Log Reg with PCA m = {m}')
            for params in LR_param_list:
                LR.logReg_wrapper(DTR_PCA, LTR, *params, idxTrain, idxTest, triplet)
            print('-----------------------------------------------------')

            # K-fold
            LR.K_fold_LogReg(DTR, LTR, K, LR_param_list, triplet, m)

if __name__ == '__main__':
    main()