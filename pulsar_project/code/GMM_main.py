import utils as u
import feature_utils as f
import plotting
import GMM
import numpy as np
import matplotlib.pyplot as plt

CSF_type_list = [(False, False), (False, True),(True, False), (True, True)] # (Tied, Diag) flags

PCA_list = [None, 7]

n_splits = 4

application_points = [(0.5, 1, 1),(0.1, 1, 1), (0.9, 1, 1)] #,

plot = True # To properly plot, PCA_list must contain 2 elements only!

def main():

    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')

    # Reduced dataset (less samples) for testing only
    # DTR, LTR = u.reduced_dataset(DTR, LTR, 200, seed=0)
    # DTE, LTE = u.reduced_dataset(DTE, LTE, 200, seed=0)

    # ---------------------- GMM classifiers ----------------------

    # priorP = u.vcol(np.array([0.5, 0.5]))
    k = 2 # Number of classes

    n = 4 # Single-Fold value
    K = 5 # K-Fold cross-validation K -> Leave-One-Out if equal to D.shape[1] (number of samples)

    delta = 1e-6
    alpha = 0.1
    psi = 0.01

    colors = ['red', 'green']

    D_merged, L_merged, idxTR_merged, idxTE_merged = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split
    idxTrain_s, idxTest_s = u.split_db_n_to_1(DTR, n) # Single-fold split

    for triplet in application_points:
        print('\nApplication point (pi_eff: {}, C_fn: {}, C_fp: {})'.format(*triplet))
        print('****************************************************')
        if plot:
            dcf_min_single = [] 
            dcf_min_kfold = []
        for i, m in enumerate(PCA_list):
            if plot:
                dcf_min_single.append([]) # Append empty list that will contain np arrays for the different models
                dcf_min_kfold.append([])
            # ----------------- Using validation set (single fold or K-fold) ----------------------
            if m is not None:
                DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain_s, idxTest_s)[0][0] # Retrieve single fold train subset
                PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
                DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PCA

            print('Single Fold ({}-to-1) GMM classifiers {}'.format(n, '(no PCA)' if m is None else f'(PCA m = {m})'))
            for j, tied_diag_pair in enumerate(CSF_type_list):
                 dcf_min_single[i].append(np.array([]))
                 for n in range(1, n_splits + 1):
                     dcf_min = GMM.GMM_wrapper(DTR if m is None else DTR_PCA, LTR, k, idxTrain_s, idxTest_s, 
                         delta, alpha, psi, n, *tied_diag_pair, triplet, single_fold=True)
                     if plot:
                         dcf_min_single[i][j] = np.append(dcf_min_single[i][j], dcf_min)
            print('-----------------------------------------------------')

            # K-fold
            print('{}-Fold cross-validation GMM classifiers {}'.format(K, '(no PCA)' if m is None else f'(PCA m = {m})')) 
            for j, tied_diag_pair in enumerate(CSF_type_list):
                if plot:
                    dcf_min_kfold[i].append(np.array([]))
                for n in range(1, n_splits + 1):
                    dcf_min = GMM.K_fold_GMM(DTR, LTR, k, K, delta, alpha, psi, n, *tied_diag_pair, triplet, m, show=True)
                    if plot:
                        dcf_min_kfold[i][j] = np.append(dcf_min_kfold[i][j], dcf_min)
#
            print('-----------------------------------------------------')

            # ------------------ Using whole Train.txt dataset and classifying Test.txt (last thing to do) ----------------
            #if m is not None:
            #      DTR_PCA_fold = u.split_dataset(D_merged, L_merged, idxTR_merged, idxTE_merged)[0][0]
            #      PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over training subset
            #      D_merged_PCA = np.dot(PCA_Proj.T, D_merged) # Project both training and validation subsets with the output of the PCA
            #print('GMM classifiers on whole dataset {}'.format('(no PCA)' if m is None else f'(PCA m = {m})'))
            #for tied_diag_pair in CSF_type_list:
            #      for n in range(1, n_splits + 1):
            #          GMM.GMM_wrapper(D_merged if m is None else D_merged_PCA, L_merged, k, idxTR_merged, idxTE_merged, 
            #              delta, alpha, psi, n, *tied_diag_pair, triplet, single_fold=True, show=True)
            #print('-----------------------------------------------------')

        if plot:
            # plotting.plotGMM(n_splits, dcf_min_single, triplet[0], CSF_type_list, colors, PCA_list)
            plotting.plotGMM(n_splits, dcf_min_kfold, triplet[0], CSF_type_list, colors, PCA_list)
    if plot:
        plt.show()

if __name__  == '__main__':
    main()
    
