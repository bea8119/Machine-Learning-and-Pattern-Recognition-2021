import utils as u
import feature_utils as f
import LogReg as LR
import DCF
import plotting as p
import numpy as np
import matplotlib.pyplot as plt

colors = ['red', 'green', 'blue']

priorT = 0.5

l_arr = np.logspace(-5, 5, 20)

printStatus = True

DTR, LTR = u.load('../data/Train.txt')
DTE, LTE = u.load('../data/Test.txt')

# Reduced dataset (less samples) for testing only
# DTR, LTR = u.reduced_dataset(DTR, LTR, 150, seed=0)
# DTE, LTE = u.reduced_dataset(DTE, LTE, 150, seed=0)

D_merged, L_merged, idxTrain, idxTest = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split for using full dataset

application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

def lambdaTuning(quad=False, saveFig=False):
    minDCF_arr = []
    for i in range(len(application_points)):
        minDCF_arr.append(np.array([]))
    for i, l in enumerate(l_arr):
        if printStatus:
            print(f'Iteration of lambda: {i + 1}')
        scores = LR.logReg_wrapper(D_merged, L_merged, l, priorT, idxTrain, idxTest, application_points[0], single_fold=False, show=False, quad=quad)
        if printStatus:
            print(f'Done training. Calculating minDCFs...')
        for i, triplet in enumerate(application_points):
            minDCF_arr[i] = np.append(minDCF_arr[i],
                DCF.DCF_unnormalized_normalized_min_binary(scores, LTE, triplet)[2]
            )
    p.plotDCFmin_vs_lambda_eval(l_arr, minDCF_arr, priorT, colors, application_points, quad=quad, saveFig=saveFig)

def main():

    # --------------- Linear LogReg, tune lambda in all application points ----------------
    lambdaTuning(quad=False, saveFig=True)
    # --------------- Quadratic LogReg, tune lambda in all application points ----------------
    lambdaTuning(quad=True, saveFig=True)

if __name__ == '__main__':
    main()