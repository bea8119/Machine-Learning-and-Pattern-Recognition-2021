import utils as u
import feature_utils as f
import MVG
import LogReg as LR
import numpy as np
import scipy.optimize

def main():
    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')
    
    application_points = [(0.5, 1, 1)]#, (0.1, 1, 1), (0.9, 1, 1)]

    # ----------------- Logistic Regression -------------------

    n = 4 # Single-Fold value
    K = 5 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)
    
    l = 1e-5 # Lambda hyperparameter
    priorT = 0.5

    for triplet in application_points:
        # ----------------- Using validation set (single fold or K-fold) ----------------------

        # Single Fold
        print('\nApplication point (pi_eff: {}, C_fn: {}, C_fp: {})'.format(*triplet))
        print('****************************************************')

        idxTrain, idxTest = u.split_db_n_to_1(DTR, n)
        print(f'Single fold ({n}-to-1) (Linear Log-Reg) (No PCA)')
        
        LR.logReg_wrapper(DTR, LTR, l, priorT, idxTrain, idxTest, triplet)
        print('-----------------------------------------------------')

    

if __name__ == '__main__':
    main()