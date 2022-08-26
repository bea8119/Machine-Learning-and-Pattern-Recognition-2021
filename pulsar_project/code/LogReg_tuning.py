from queue import PriorityQueue
import utils as u
import feature_utils as f
import LogReg as LR
import numpy as np
import matplotlib.pyplot as plt

PCA_list = [None, 7, 6]
colors = ['red', 'green', 'blue']

def main():

    DTR, LTR = u.load('../data/Train.txt')

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    priorT = 0.5
    n = 4 # Single fold value
    K = 5 # K-fold value
    l_arr = np.logspace(-5, 5, 10)

    idxTrain, idxTest = u.split_db_n_to_1(DTR, n) # Single-fold split

    for m in PCA_list:

        fig_single = plt.figure('Single-fold lambda tuning {}'.format('(no PCA)' if m is None else f'(PCA m = {m})'))
        fig_kfold = plt.figure('K-fold lambda tuning {}'.format('(no PCA)' if m is None else f'(PCA m = {m})'))

        if m is not None:
            DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
            PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
            DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PC

        for i, triplet in enumerate(application_points):
            c = colors[i]
            min_DCF_single = np.array([])
            min_DCF_kfold = np.array([])
            for l in l_arr:
                # ----------------- Using validation set (single fold or K-fold) ----------------------

                # Single Fold
                
                min_DCF_single = np.append(min_DCF_single, 
                    LR.logReg_wrapper(DTR if m is None else DTR_PCA, LTR, l, priorT, idxTrain, idxTest, triplet, show=False)
                )

                # # K-fold
                # min_DCF_kfold = np.concatenate((min_DCF_kfold, 
                #     LR.K_fold_LogReg(DTR, LTR, K, [(l, priorT)], triplet, m, show=False))) 

            plt.figure(fig_single) # Set fig_single as active figure
            plt.plot(l_arr, min_DCF_single, color=c, label=f'eff_prior = {triplet[0]}')

        plt.figure(fig_single)
        plt.xlim([min(l_arr), max(l_arr)])
        plt.xscale('log')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('min DCF')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

if __name__ == '__main__':
    main()