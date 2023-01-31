import utils as u
import feature_utils as f
import SVM
import plotting as p
import numpy as np
import matplotlib.pyplot as plt

PCA_list = [None]
colors = ['red', 'green', 'blue', 'orange']

priorT_b = [None]
K_svm = 1 # Value of sqrt(psi)

c_list = [1, 5, 10, 20]
gamma_list = [1e-2, 1e-1, 1e-0, 1e1]

printStatus = True

def main():

    DTR, LTR = u.load('../data/Train.txt')

    # Reduced dataset (less samples) for testing only
    # DTR, LTR = u.reduced_dataset(DTR, LTR, 150, seed=0)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    d = 2 # degree of Polynomial kernel
    n = 4 # Single fold value
    K = 5 # K-fold value
    C_arr = np.logspace(-5, 5, 20)

    idxTrain, idxTest = u.split_db_n_to_1(DTR, n) # Single-fold split

    for pi_b in priorT_b:
        print('pi_T = {}'.format(pi_b))
        for m in PCA_list:
            print('PCA m = {}'.format(m))
            if m is not None:
                DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
                PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
                DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PC

            # --------------- Linear SVM, tune C in all application points ----------------
            min_DCF_single = []
            min_DCF_kfold = []
            for i, triplet in enumerate(application_points):
                print('APP POINT: {}'.format(triplet))
                min_DCF_single.append(np.array([]))
                min_DCF_kfold.append(np.array([]))
                for C in C_arr:
                    # ----------------- Using validation set (single fold or K-fold) ----------------------
                    # # Single Fold
                    # print('\tSingle Fold')
                    # min_DCF_single[i] = np.append(min_DCF_single[i], 
                    #     SVM.SVM_wrapper(DTR if m is None else DTR_PCA, LTR, K_svm, C, pi_b, idxTrain, idxTest, triplet, show=False, kern=False)
                    # )

                    print('\tKfold')
                    # K-fold
                    min_DCF_kfold[i] = np.append(min_DCF_kfold[i], 
                        SVM.K_fold_SVM(DTR, LTR, K, K_svm, C, pi_b, triplet, m, show=False, kern=False, printStatus=True)
                    )

            p.plotDCFmin_vs_C_linearSVM(C_arr, None, min_DCF_kfold, pi_b, m, n, K, colors, application_points)

    # plt.show()
    
    # ------------ Quadratic kernel SVM, tune C and c jointly, same application point (0.5, 1, 1) (unbalanced) ----
    for m in PCA_list:
        min_DCF_single = []
        min_DCF_kfold = []
        print('PCA m = {}'.format(m))

        if m is not None:
            DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
            PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
            DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PC

    
        for i, c in enumerate(c_list):
            print('c = {}'.format(c))
            min_DCF_single.append(np.array([]))
            min_DCF_kfold.append(np.array([]))
            for C in C_arr:
                # ----------------- Using validation set (single fold or K-fold) ----------------------
                # # Single Fold
                # min_DCF_single[i] = np.append(min_DCF_single[i], 
                #     SVM.SVM_wrapper(DTR if m is None else DTR_PCA, LTR, K_svm, C, priorT_b[0], idxTrain, idxTest, application_points[0], show=False, kern=True, c=c, d=d)
                # )

                # K-fold
                print('\tKfold')
                min_DCF_kfold[i] = np.append(min_DCF_kfold[i], 
                    SVM.K_fold_SVM(DTR, LTR, K, K_svm, C, priorT_b[0], application_points[0], m, show=True, kern=True, c=c, d=d, printStatus=True)
                )
        p.plotDCFmin_vs_C_quadSVM(C_arr, None, min_DCF_kfold, m, n, K, colors, application_points[0], c_list)
    # plt.show()

    # ------------ RBF kernel SVM, tune C and gamma jointly, same appplication point (0.5, 1, 1) (unbalanced) ------
    for m in PCA_list:
        min_DCF_single = []
        min_DCF_kfold = []
        print('PCA m = {}'.format(m))
        if m is not None:
            DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
            PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
            DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PC

        for i, gamma in enumerate(gamma_list):
            print('gamma = {}'.format(gamma))
            min_DCF_single.append(np.array([]))
            min_DCF_kfold.append(np.array([]))
            for C in C_arr:
                # ----------------- Using validation set (single fold or K-fold) ----------------------
                # # Single Fold
                # print('\tSingle Fold')
                # min_DCF_single[i] = np.append(min_DCF_single[i], 
                #     SVM.SVM_wrapper(DTR if m is None else DTR_PCA, LTR, K_svm, C, priorT_b[0], idxTrain, idxTest, application_points[0], show=False, kern=True, gamma=gamma, Poly_RBF=False)
                # )

                # K-fold
                print('\tKfold')
                min_DCF_kfold[i] = np.append(min_DCF_kfold[i], 
                    SVM.K_fold_SVM(DTR, LTR, K, K_svm, C, priorT_b[0], application_points[0], m, show=True, kern=True, gamma=gamma, Poly_RBF=False, printStatus=True)
                )
        p.plotDCFmin_vs_C_RBFSVM(C_arr, None, min_DCF_kfold, m, n, K, colors, application_points[0], gamma_list)

    plt.show()

if __name__ == '__main__':
    main()