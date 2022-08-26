from utils import vcol, split_dataset
from feature_utils import PCA_givenM, Z_normalization
from DCF import DCF_unnormalized_normalized_min_binary
import numpy as np
import scipy.optimize

# def lambdaTuning()

# def computeAccuracy_logreg_binary(scoreArray, TrueL):
#     PredictedL = np.array([(1 if score > 0 else 0) for score in scoreArray.ravel()])
#     NCorrect = (PredictedL.ravel() == TrueL.ravel()).sum() # Will count as 1 the "True"
#     NTotal = TrueL.size
#     return float(NCorrect) / float(NTotal)

def logReg_wrapper(D, L, l, priorT, idxTrain, idxTest, triplet, single_fold=True):

    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, idxTrain, idxTest)

    # Apply Z-normalization from training, apply the same transformation on the test set
    DTR, mean, std = Z_normalization(DTR)
    DTE = Z_normalization(DTE, mean, std)

    logRegObj = logRegClass(DTR, LTR, l, priorT)
    llrs = logRegObj.logreg_llrs(DTE)
    if single_fold:
        testDCF_LogReg(LTE, l, priorT, llrs, triplet)
        return
    return llrs

def testDCF_LogReg(LTE, l, priorT, llrs, triplet):
    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(llrs, LTE, triplet)
    print(f'\tLinear LogReg (lambda = {l}, priorT = {priorT}) -> min DCF: {round(dcf_min, 3)}')

class logRegClass:
    def __init__(self, DTR, LTR, l, priorT):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.priorT = priorT

    def logreg_obj_binary(self, v):
        w, b = vcol(v[0:-1]), v[-1]
        s0 = np.dot(w.T, self.DTR[:, self.LTR == 0]) + b
        s1 = np.dot(w.T, self.DTR[:, self.LTR == 1]) + b
        # z = 2.0 * self.LTR - 1 # Encoding to +1 and -1

        # Cross-entropy
        mean_term1 = np.logaddexp(0, -s1).mean() # directly z=1 since it is the true samples
        mean_term0 = np.logaddexp(0, s0).mean()
        crossEntropy = self.priorT * mean_term1 + (1 - self.priorT) * mean_term0
        return 0.5 * self.l * np.linalg.norm(w)**2 + crossEntropy

    def logreg_llrs(self, DTE):
        x0 = np.zeros(self.DTR.shape[0] + 1)
        self.DTR, means, std = Z_normalization(self.DTR) # Z-normalization
        (v, J, d) = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj_binary, x0, approx_grad=True)
        w = vcol(v[0:-1])
        b = v[-1]
        DTE = Z_normalization(DTE, means, std) # Same transformation on Test set
        llrs = np.dot(w.T, DTE) + b # Posterior log-likelihood ratio
        return llrs.ravel()


def K_fold_LogReg(D, L, K, LR_param_list, app_triplet, PCA_m=None, seed=0):
    if PCA_m is not None:
        msg = f' with PCA m = {PCA_m}'
    else: 
        msg = ' (no PCA)'
    print(f'{K}-Fold cross-validation Linear Log Reg{msg}')

    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) 

    for params in LR_param_list:
        startTest = 0
        # For DCF computation
        llrs_all = np.array([])
        for j in range(K):
            idxTest = idx[startTest: (startTest + nTest)]
            idxTrain = np.setdiff1d(idx, idxTest)
            if PCA_m is not None:
                DTR_PCA_fold = split_dataset(D, L, idxTrain, idxTest)[0][0]
                PCA_P = PCA_givenM(DTR_PCA_fold, PCA_m)
                D_PCA = np.dot(PCA_P.T, D)
                llrs = logReg_wrapper(D_PCA, L, *params, idxTrain, idxTest, app_triplet, single_fold=False)
            else:
                llrs = logReg_wrapper(D, L, *params, idxTrain, idxTest, app_triplet, single_fold=False)
            llrs_all = np.concatenate((llrs_all, llrs))
            startTest += nTest
        
        # DCF computation (compute)
        trueL_ordered = L[idx] # idx was computed randomly before
        (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(llrs_all, trueL_ordered, app_triplet)
        print(f'\tLinear Log Reg (lambda = {params[0]}, priorT = {params[1]}) -> min DCF: {round(dcf_min, 3)}')