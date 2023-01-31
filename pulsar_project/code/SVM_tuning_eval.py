import utils as u
import feature_utils as f
import SVM
import DCF
import plotting as p
import numpy as np
import matplotlib.pyplot as plt

colors = ['red', 'green', 'blue', 'orange']

priorT_b = None
K_svm = 1 # Value of sqrt(psi)

c_list = [1, 5, 10]
gamma_list = [1e-2, 1e-1, 1e-0]

C_arr = np.logspace(-5, 5, 15)

printStatus = True

DTR, LTR = u.load('../data/Train.txt')
DTE, LTE = u.load('../data/Test.txt')

# Reduced dataset (less samples) for testing only
# DTR, LTR = u.reduced_dataset(DTR, LTR, 3000, seed=0)
# DTE, LTE = u.reduced_dataset(DTE, LTE, 3000, seed=0)

D_merged, L_merged, idxTrain, idxTest = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split for using full dataset

application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

d = 2 # degree of Polynomial kernel

def main():

    # # --------------- Linear SVM, tune C in all application points ----------------
    # minDCF_arr = []
    # for i in range(len(application_points)):
    #     minDCF_arr.append(np.array([]))
    # for i, C in enumerate(C_arr):
    #     if printStatus:
    #         print(f'Iteration of C: {i + 1}')
    #     scores = SVM.SVM_wrapper(D_merged, L_merged, K_svm, C, priorT_b, idxTrain, idxTest, application_points[0], single_fold=False, show=False)
    #     if printStatus:
    #         print(f'Done training. Calculating minDCFs...')
    #     for i, triplet in enumerate(application_points):
    #         minDCF_arr[i] = np.append(minDCF_arr[i],
    #             DCF.DCF_unnormalized_normalized_min_binary(scores, LTE, triplet)[2]
    #         )
    # if printStatus:
    #     print('Finished tuning!\n')
    # p.plotDCFmin_vs_C_linearSVM_eval(C_arr, minDCF_arr, priorT_b, colors, application_points, saveFig=True)
    
    # ------------ Quadratic kernel SVM, tune C and c jointly, same application point (0.5, 1, 1) (unbalanced) ----
    # minDCF_arr = []
    # for j, c in enumerate(c_list):
    #     minDCF_arr.append(np.array([]))
    #     if printStatus:
    #         print(f'Iteration of c: {j + 1}')
    #     for i, C in enumerate(C_arr):
    #         if printStatus:
    #             print(f'\tIteration of C: {i + 1}')
    #         scores = SVM.SVM_wrapper(D_merged, L_merged, K_svm, C, priorT_b, idxTrain, idxTest, application_points[0], c, d, single_fold=False, show=False, kern=True, Poly_RBF=True)
    #         if printStatus:
    #             print(f'\tDone training. Calculating minDCFs...')
    #         minDCF_arr[j] = np.append(minDCF_arr[j], DCF.DCF_unnormalized_normalized_min_binary(scores, LTE, application_points[0])[2])
    # if printStatus:
    #     print('Finished tuning!\n')
    # p.plotDCFmin_vs_C_quadSVM_eval(C_arr, minDCF_arr, priorT_b, colors, application_points[0], c_list, saveFig=True)

    # ------------ RBF kernel SVM, tune C and gamma jointly, same appplication point (0.5, 1, 1) (unbalanced) ------
    minDCF_arr = []
    for j, gamma in enumerate(gamma_list):
        minDCF_arr.append(np.array([]))
        if printStatus:
            print(f'Iteration of gamma: {j + 1}')
        for i, C in enumerate(C_arr):
            if printStatus:
                print(f'\tIteration of C: {i + 1}')
            scores = SVM.SVM_wrapper(D_merged, L_merged, K_svm, C, priorT_b, idxTrain, idxTest, application_points[0], gamma=gamma, single_fold=False, show=False, kern=True, Poly_RBF=False)
            if printStatus:
                print(f'\tDone training. Calculating minDCFs...')
            minDCF_arr[j] = np.append(minDCF_arr[j], DCF.DCF_unnormalized_normalized_min_binary(scores, LTE, application_points[0])[2])
    if printStatus:
        print('Finished tuning!\n')
    p.plotDCFmin_vs_C_RBFSVM_eval(C_arr, minDCF_arr, priorT_b, colors, application_points[0], gamma_list, saveFig=True)

if __name__ == '__main__':
    main()