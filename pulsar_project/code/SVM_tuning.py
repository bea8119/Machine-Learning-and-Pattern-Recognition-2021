import utils as u
import feature_utils as f
import SVM
import plotting as p
import numpy as np
import matplotlib.pyplot as plt

PCA_list = [None, 7]
colors = ['red', 'green', 'blue', 'orange', 'black']

kernel_SVM = True # False for Linear SVM, regardless of the next flag
Poly_RBF = True # True for polynomial, False for RBF kernel SVM (assuming kernel flag == True)
K_svm = 1 # Value of sqrt(psi)

c_list = [1, 5, 10, 20, 30]
gamma_list = [1e-2, 1e-1, 1e-0, 1e1, 1e2]

def main():

    DTR, LTR = u.load('../data/Train.txt')

    # Reduced dataset (less samples) for testing only
    # DTR, LTR = u.reduced_dataset(DTR, LTR, 500, seed=0)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    priorT_b = None # Test on main application without class balancing
    d = 2 # degree of Polynomial kernel
    n = 4 # Single fold value
    K = 5 # K-fold value
    C_arr = np.logspace(-5, 5)

    idxTrain, idxTest = u.split_db_n_to_1(DTR, n) # Single-fold split

    for m in PCA_list:
        if m is not None:
            DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
            PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
            DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PC


        # --------------- Linear SVM, tune C in all application points ----------------
        min_DCF_single = []
        min_DCF_kfold = []
        for i, triplet in enumerate(application_points):
            min_DCF_single.append(np.array([]))
            min_DCF_kfold.append(np.array([]))
            for C in C_arr:
                # ----------------- Using validation set (single fold or K-fold) ----------------------
                # Single Fold
                min_DCF_single[i] = np.append(min_DCF_single[i], 
                    SVM.SVM_wrapper(DTR if m is None else DTR_PCA, LTR, K_svm, C, priorT_b, idxTrain, idxTest, triplet, show=False, kern=False)
                )

                # K-fold
                min_DCF_kfold[i] = np.append(min_DCF_kfold[i], 
                    SVM.K_fold_SVM(DTR, LTR, K, K_svm, C, priorT_b, triplet, m, show=False, kern=False)
                )
        p.plotDCFmin_vs_C_linearSVM(C_arr, min_DCF_single, min_DCF_kfold, m, n, K, colors, application_points)

        # ------------ Quadratic kernel SVM, tune C and c jointly on the same appplication point (0.5, 1, 1) ---------------
        min_DCF_single = []
        min_DCF_kfold = []
        for i, c in enumerate(c_list):
            min_DCF_single.append(np.array([]))
            min_DCF_kfold.append(np.array([]))
            for C in C_arr:
                # ----------------- Using validation set (single fold or K-fold) ----------------------
                # Single Fold
                min_DCF_single[i] = np.append(min_DCF_single[i], 
                    SVM.SVM_wrapper(DTR if m is None else DTR_PCA, LTR, K_svm, C, priorT_b, idxTrain, idxTest, triplet, show=False, kern=True, c=c, d=d)
                )

                # K-fold
                min_DCF_kfold[i] = np.append(min_DCF_kfold[i], 
                    SVM.K_fold_SVM(DTR, LTR, K, K_svm, C, priorT_b, triplet, m, show=False, kern=True, c=c, d=d)
                )
        p.plotDCFmin_vs_C_quadSVM(C_arr, min_DCF_single, min_DCF_kfold, m, n, K, colors, application_points[0], c_list)

        # ------------ RBF kernel SVM, tune C and gamma jointly on the same appplication point (0.5, 1, 1) ---------------
        min_DCF_single = []
        min_DCF_kfold = []
        for i, gamma in enumerate(gamma_list):
            min_DCF_single.append(np.array([]))
            min_DCF_kfold.append(np.array([]))
            for C in C_arr:
                # ----------------- Using validation set (single fold or K-fold) ----------------------
                # Single Fold
                min_DCF_single[i] = np.append(min_DCF_single[i], 
                    SVM.SVM_wrapper(DTR if m is None else DTR_PCA, LTR, K_svm, C, priorT_b, idxTrain, idxTest, triplet, show=False, kern=True, gamma=gamma, Poly_RBF=False)
                )

                # K-fold
                min_DCF_kfold[i] = np.append(min_DCF_kfold[i], 
                    SVM.K_fold_SVM(DTR, LTR, K, K_svm, C, priorT_b, triplet, m, show=False, kern=True, gamma=gamma, Poly_RBF=False)
                )
        p.plotDCFmin_vs_C_RBFSVM(C_arr, min_DCF_single, min_DCF_kfold, m, n, K, colors, application_points[0], gamma_list)

    plt.show()

if __name__ == '__main__':
    main()