import utils as u
import feature_utils as f
import LogReg as LR
import plotting as p
import numpy as np
import matplotlib.pyplot as plt

PCA_list = [None, 7]
colors = ['red', 'green', 'blue']

quadratic = True # False for Linear Logistic Regression
printStatus = True
save_fig = True

def main():

    DTR, LTR = u.load('../data/Train.txt')

    # Reduced dataset (less samples) for testing only
    # DTR, LTR = u.reduced_dataset(DTR, LTR, 500, seed=0)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    priorT = 0.5
    n = 4 # Single fold value
    K = 5 # K-fold value
    l_arr = np.logspace(-5, 5, 40)

    idxTrain, idxTest = u.split_db_n_to_1(DTR, n) # Single-fold split

    if printStatus:
        total_steps = 1 * len(application_points) * len(PCA_list) * len(l_arr)
        step = 1

    for m in PCA_list:
        if m is not None:
            DTR_PCA_fold = u.split_dataset(DTR, LTR, idxTrain, idxTest)[0][0] # Retrieve single fold train subset
            PCA_Proj = f.PCA_givenM(DTR_PCA_fold, m) # Apply PCA over Training subset
            DTR_PCA = np.dot(PCA_Proj.T, DTR) # Project both training and validation subsets with the output of the PC

        min_DCF_single = []
        min_DCF_kfold = []

        for i, triplet in enumerate(application_points):
            min_DCF_single.append(np.array([]))
            min_DCF_kfold.append(np.array([]))
            for l in l_arr:
                # ----------------- Using validation set (single fold or K-fold) ----------------------

                # Single Fold

                if printStatus:
                    print(f'Step {step} (single-fold) of {total_steps}')

                min_DCF_single[i] = np.append(min_DCF_single[i], 
                    LR.logReg_wrapper(DTR if m is None else DTR_PCA, LTR, l, priorT, idxTrain, idxTest, triplet, show=False, quad=quadratic)
                )

                if printStatus:
                    # step += 1
                    print(f'Step {step} (K-fold) of {total_steps}')

                # K-fold (Takes very long)
                min_DCF_kfold[i] = np.append(min_DCF_kfold[i], 
                    LR.K_fold_LogReg(DTR, LTR, K, l, priorT, triplet, m, show=False, quad=quadratic, printStatus=printStatus)
                )

                if printStatus:
                    step += 1

        p.plotDCFmin_vs_lambda(l_arr, None, min_DCF_kfold, m, n, K, colors, application_points, quad=quadratic, save_fig=save_fig)

    plt.show()

if __name__ == '__main__':
    main()