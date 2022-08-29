from utils import vcol, split_dataset
from feature_utils import PCA_givenM, Z_normalization
from DCF import DCF_unnormalized_normalized_min_binary
import numpy as np
import scipy.optimize

# def computeAccuracy_logreg_binary(scoreArray, TrueL):
#     PredictedL = np.array([(1 if score > 0 else 0) for score in scoreArray.ravel()])
#     NCorrect = (PredictedL.ravel() == TrueL.ravel()).sum() # Will count as 1 the "True"
#     NTotal = TrueL.size
#     return float(NCorrect) / float(NTotal)

def vec_xxT(x):
    '''Receives a 1D array, returns 1D array after computing product and reordering column-wise'''
    x = vcol(x)
    x_xT = np.dot(x, x.T) # Output will have D x D size (D is original dimension of feature space)
    vec_result = np.reshape(x_xT, -1, order='F') # It is a 1D array
    return vec_result

def expand_f_space(D):
    exp_D = np.apply_along_axis(vec_xxT, 0, D)
    exp_D = np.vstack([exp_D, D]) # Stack original D at the bottom
    return exp_D

def logReg_wrapper(D, L, l, priorT, idxTrain, idxTest, triplet, single_fold=True, show=True, quad=False):

    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, idxTrain, idxTest)

    # Apply Z-normalization on the training set (of the current fold), apply the same transformation on the test set
    DTR, mean, std = Z_normalization(DTR)
    DTE = Z_normalization(DTE, mean, std)

    if quad:
        DTR = expand_f_space(DTR)
        DTE = expand_f_space(DTE)

    logRegObj = logRegClass(DTR, LTR, l, priorT)
    llrs = logRegObj.logreg_llrs(DTE)
    if single_fold:
        return minDCF_LogReg(LTE, l, priorT, llrs, triplet, show=show, quad=quad)
    return llrs

def minDCF_LogReg(LTE, l, priorT, llrs, triplet, show=True, quad=False):
    '''Returns DCF min for lambda tuning'''
    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(llrs, LTE, triplet)
    if show:
        print('\t{} Log Reg (lambda = {}, priorT = {}) -> min DCF: {}'.format(
            'Quadratic' if quad else 'Linear', l, priorT, round(dcf_min, 3)))
    return dcf_min

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

        # Cross-entropy
        mean_term1 = np.logaddexp(0, -s1).mean() # directly z=1 since it is the true samples
        mean_term0 = np.logaddexp(0, s0).mean()
        crossEntropy = self.priorT * mean_term1 + (1 - self.priorT) * mean_term0
        return 0.5 * self.l * np.linalg.norm(w)**2 + crossEntropy

    def logreg_llrs(self, DTE):
        x0 = np.zeros(self.DTR.shape[0] + 1)
        (v, J, d) = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj_binary, x0, approx_grad=True)
        w = vcol(v[0:-1])
        b = v[-1]
        p_lprs = np.dot(w.T, DTE) + b # Posterior log-probability ratio
        llrs = p_lprs - np.log(self.priorT / (1 - self.priorT)) # Unplug the prior probabilities to have only the log-likelihood ratios
        return llrs.ravel()

def K_fold_LogReg(D, L, K, l, priorT, app_triplet, PCA_m=None, seed=0, show=True, quad=False):
    if PCA_m is not None:
        msg = f'with PCA m = {PCA_m}'
    else: 
        msg = '(no PCA)'
    if show:
        print('{}-Fold cross-validation {} Log Reg {}'.format(K, 'Quadratic' if quad else 'Linear', msg))

    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) 

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
            llrs = logReg_wrapper(D_PCA, L, l, priorT, idxTrain, idxTest, app_triplet, single_fold=False, show=show, quad=quad)
        else:
            llrs = logReg_wrapper(D, L, l, priorT, idxTrain, idxTest, app_triplet, single_fold=False, show=show, quad=quad)

        llrs_all = np.concatenate((llrs_all, llrs))
        startTest += nTest
        
    # DCF computation (compute)
    trueL_ordered = L[idx] # idx was computed randomly before
    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(llrs_all, trueL_ordered, app_triplet)
    if show:
        print('\t{} LogReg (lambda = {}, priorT = {}) -> min DCF: {}'.format(
            'Quadratic' if quad else 'Linear', l, priorT, round(dcf_min, 3)))
    return dcf_min