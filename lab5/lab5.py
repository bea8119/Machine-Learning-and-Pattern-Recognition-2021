import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.special

# sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from lab3.lab3 import datasetCovarianceM 
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

def main():
    D, L = load('../datasets/iris.csv')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L) # DTRain DTEst

    mu_arr = [vcol(DTR[:, LTR == i].mean(axis=1)) for i in range(3)]
    C_arr = [datasetCovarianceM(DTR[:, LTR == i]) for i in range(3)]

    # now focus on the TEST samples
    N_test = DTE.shape[1]

    priorP = vcol(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
    S = np.zeros((3, N_test))

    for i in range(3):
        # S will store the LOG densities
        S[i, :] = vrow(np.array([logpdf_GAU_ND(vcol(DTE[:, j]), mu_arr[i], C_arr[i]) for j in range(N_test)]))
    print(S.shape)

    # Work without LOG densities
#  ---------------------------------------
    SJoint = np.exp(S) * priorP
    SJoint_Sol = np.load('./sol/SJoint_MVG.npy')

    SMarginal = vrow(SJoint.sum(0)) # row vector storing the sum over rows (classes) of each sample

    SPost = SJoint / SMarginal # in here we have for each column (sample), a row of scores (one per class)

    PredictedL = SPost.argmax(0) # INDEX of max through axis 0

    CorrectPred = (PredictedL == LTE) # Boolean array of predictions
    CorrectCount = sum(CorrectPred)
    accuracy = CorrectCount / N_test
    error_rate = 1 - accuracy
    print(accuracy * 100)
# ---------------------------------------

# Now, with log densities
# ---------------------------------------
    logSJoint = S + np.log(priorP)

    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0)) # compute for each column

    logSPost = logSJoint - logSMarginal

    SPost_afterLog = np.exp(logSPost)

    PredictedL_log = SPost_afterLog.argmax(0) # INDEX of max through axis 0
    CorrectPredLog = (PredictedL_log == LTE)
    print(CorrectPredLog)

if __name__ == "__main__":
    main()

