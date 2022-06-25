import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.special

sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
# sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from lab3.lab3 import SW_compute, datasetCovarianceM, SB_compute
from utility.vrow_vcol import vcol, vrow
from lab2.load_plot import load
from lab4.lab4 import logpdf_GAU_ND

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) # should be 50
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) # take a random order of indexes from 0 to N
    idxTrain = idx[0:nTrain] 
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

# This function will return the test samples, mean and covariance matrix per class
# Dataset, Labels, prior probabilities (to be passed in vcol shape), k number of classes
def classifierSetup(D, L, k, splitFunction, tied=False): 
    # Gaussian Classifier
    (DTR, LTR), (DTE, LTE) = splitFunction(D, L) # DTRain DTEst

    mu_arr = [vcol(DTR[:, LTR == i].mean(axis=1)) for i in range(k)]

    if tied == False:
        C_arr = [datasetCovarianceM(DTR[:, LTR == i]) for i in range(k)]
    else:
        C_tied = SW_compute(DTR, LTR, k)
        C_arr = [C_tied for i in range(k)]
    return (DTE, LTE), mu_arr, C_arr 

def testLabelPredAccuracy(PredictedL, LTE, classifierName):
    CorrectPred = (PredictedL == LTE) # Boolean array of predictions
    CorrectCount = sum(CorrectPred)
    accuracy = CorrectCount / LTE.shape[0]
    error_rate = 1 - accuracy
    print('Accuracy of' + classifierName + ' Gaussian classifier: ' + str(round(accuracy * 100, 2)) + '%')
    print('Error rate of' + classifierName + ' Gaussian classifier: ' + str(round(error_rate * 100, 2)) + '%')


def gaussianCSF(DTE, LTE, k, mu_arr, C_arr, priorP, CSF_name, log=False):
     # now focus on the TEST samples
    N_test = DTE.shape[1]

    S = np.zeros((k, N_test))

    for i in range(k):
        # S will store the LOG densities
        S[i, :] = vrow(np.array([logpdf_GAU_ND(vcol(DTE[:, j]), mu_arr[i], C_arr[i]) for j in range(N_test)]))

    if log == False:
        # Work WITHOUT LOG densities
    #  ---------------------------------------
        SJoint = np.exp(S) * priorP

        SMarginal = vrow(SJoint.sum(0)) # row vector storing the sum over rows (classes) of each sample

        SPost = SJoint / SMarginal # in here we have for each column (sample), a row of scores (one per class)

        PredictedL = SPost.argmax(0) # INDEX of max through axis 0

        testLabelPredAccuracy(PredictedL, LTE, CSF_name)

        return PredictedL
# ---------------------------------------
    else:
        # Now, with log densities
    # ---------------------------------------
        logSJoint = S + np.log(priorP)

        logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0)) # compute for each column

        logSPost = logSJoint - logSMarginal

        SPost_afterLog = np.exp(logSPost)

        PredictedL_log = SPost_afterLog.argmax(0) # INDEX of max through axis 0
        
        CSF_name = CSF_name + ' with log'
        testLabelPredAccuracy(PredictedL_log, LTE, CSF_name)

        return PredictedL_log

def naiveBayesGaussianCSF(DTE, LTE, k, mu_arr, C_arr, priorP, log=False):
    C_naive_arr = [C_arr[i] * np.identity(C_arr[i].shape[0]) for i in range(3)] # element by element mult.
    gaussianCSF(DTE, LTE, k, mu_arr, C_naive_arr, priorP, " Naive Bayes", log)

def tiedCovarianceGaussianCSF(D, L, k, splitFuncion, priorP, log=False):
    (DTE, LTE), mu_arr, C_arr = classifierSetup(D, L, k, splitFuncion, tied=True)
    gaussianCSF(DTE, LTE, k, mu_arr, C_arr, priorP, " Tied Covariance", log)

def main():
    D, L = load('../datasets/iris.csv')
    priorP = vcol(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
    k = 3

    (DTE, LTE), mu_arr, C_arr = classifierSetup(D, L, k, split_db_2to1)

    gaussianCSF(DTE, LTE, k, mu_arr, C_arr, priorP, "", log=True)

    naiveBayesGaussianCSF(D, L, k, mu_arr, C_arr, priorP, log=False)

    tiedCovarianceGaussianCSF(D, L, k, split_db_2to1, priorP, log=True)



if __name__ == "__main__":
    main()

