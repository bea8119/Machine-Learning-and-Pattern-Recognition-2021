from utils import vcol, vrow, split_dataset
from feature_utils import PCA_givenM, Z_normalization
from DCF import DCF_unnormalized_normalized_min_binary
import numpy as np
import scipy.optimize


def vec_xxT(x):
    '''Receives a 1D array, returns 1D array after computing product and reordering column-wise'''
    x = vcol(x)
    # Output will have D x D size (D is original dimension of feature space)
    x_xT = np.dot(x, x.T)
    vec_result = np.reshape(x_xT, -1, order='F')  # It is a 1D array
    return vec_result


def expand_f_space(D):
    exp_D = np.apply_along_axis(vec_xxT, 0, D)
    exp_D = np.vstack([exp_D, D])  # Stack original D at the bottom
    return exp_D


def logReg_wrapper(D, L, l, priorT, idxTrain, idxTest, triplet, single_fold=True, show=True, quad=False, calibrate=False):

    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, idxTrain, idxTest)

    # Apply Z-normalization on the training set (of the current fold), apply the same transformation on the test set
    DTR, mean, std = Z_normalization(DTR)
    DTE = Z_normalization(DTE, mean, std)

    if quad:
        DTR = expand_f_space(DTR)
        DTE = expand_f_space(DTE)

    logRegObj = logRegClass(DTR, LTR, l, priorT)
    scores = logRegObj.logreg_scores(DTE)

    if calibrate:
        scores, w, b = calibrate_scores(scores, LTE, 0.5)

    if single_fold:
        return testDCF_LogReg(LTE, l, priorT, scores, triplet, show=show, quad=quad)
    return scores


def testDCF_LogReg(LTE, l, priorT, scores, triplet, show=True, quad=False):
    '''Returns DCF min for lambda tuning'''
    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(
        scores, LTE, triplet)
    if show:
        print('\t{} Log Reg (lambda = {}, priorT = {}) -> min DCF: {}    act DCF: {}'.format(
            'Quadratic' if quad else 'Linear', l, priorT, round(dcf_min, 3), round(dcf_norm, 3)))
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
        # directly z=1 since it is the true samples
        mean_term1 = np.logaddexp(0, -s1).mean()
        mean_term0 = np.logaddexp(0, s0).mean()
        crossEntropy = self.priorT * mean_term1 + \
            (1 - self.priorT) * mean_term0
        return 0.5 * self.l * np.linalg.norm(w)**2 + crossEntropy

    def logreg_scores(self, DTE, calibrate=False):
        x0 = np.zeros(self.DTR.shape[0] + 1)
        (v, J, d) = scipy.optimize.fmin_l_bfgs_b(
            self.logreg_obj_binary,
            x0,
            approx_grad=True,
            factr=1.0,
            # maxiter=100,
        )
        w = vcol(v[0:-1])
        b = v[-1]
        p_lprs = np.dot(w.T, DTE) + b  # Posterior log-probability ratio
        if calibrate:
            return p_lprs.ravel(), w, b
        return p_lprs.ravel()


def K_fold_LogReg(D, L, K, l, priorT, app_triplet, PCA_m=None, seed=0, show=True, quad=False, printStatus=False, calibrate=False):
    
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    even_increase = [round(x) for x in np.linspace(0, D.shape[1], K + 1)]

    # For DCF computation
    scores_all = np.array([])
    for j in range(K):
        if printStatus:
            print('fold {} start...'.format(j + 1))
        idxTest = idx[even_increase[j]: even_increase[j + 1]]
        idxTrain = np.setdiff1d(idx, idxTest)
        if PCA_m is not None:
            DTR_PCA_fold = split_dataset(D, L, idxTrain, idxTest)[0][0]
            PCA_P = PCA_givenM(DTR_PCA_fold, PCA_m)
            D_PCA = np.dot(PCA_P.T, D)
            scores = logReg_wrapper(D_PCA, L, l, priorT, idxTrain, idxTest,
                                    app_triplet, single_fold=False, show=show, quad=quad)
        else:
            scores = logReg_wrapper(D, L, l, priorT, idxTrain, idxTest,
                                    app_triplet, single_fold=False, show=show, quad=quad)

        scores_all = np.concatenate((scores_all, scores))

    # DCF computation (compute)
    trueL_ordered = L[idx]  # idx was computed randomly before

    if calibrate:
        scores_all, w, b = calibrate_scores(scores_all, trueL_ordered, 0.5)

    if printStatus:
        print('calculating minDCF...')

    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(
        scores_all, trueL_ordered, app_triplet)
    if show:
        print('\t{} LogReg (lambda = {}, priorT = {}) -> min DCF: {}    act DCF: {}'.format(
            'Quadratic' if quad else 'Linear', l, priorT, round(dcf_min, 3), round(dcf_norm, 3)))
    return dcf_min

def calibrate_scores(scores_D, L, eff_prior, w=None, b=None):
    '''This function takes the scores (1D) of the classifiers (whatever the model yields as scores) and returns
    the calibrated scores after applying a LogReg with lambda = 0 (no regularization) and a given effective
    prior to use as the "priorT" of the LogReg model. This LogReg will return a new set of scores (applying the 
    optimal w and b to the same scores used for training the data) that are the transformed (hopefully optimal)
    scores and also returns the corresponding w, b pair. If you pass w and b, instead, the calibration will 
    happen by applying the transformation given (skipping the training)'''
    
    scores_D = vrow(scores_D)
    if w is None and b is None:
        logRegObj = logRegClass(scores_D, L, 0, eff_prior)
        logreg_scores, w, b = logRegObj.logreg_scores(scores_D, calibrate=True)
        calibrated_scores = logreg_scores - np.log(eff_prior / (1 - eff_prior))
        return calibrated_scores, w, b
    else:
        calibrated_scores = np.dot(w.T, scores_D) + b
        return calibrated_scores.ravel()


def fusionModel(csf_1_scores, csf_2_scores, L, eff_prior, w=None, b=None):
    '''Receives two 1D arrays containing the scores of the output of 2 classifiers. Trains a logistic regression 
    model for performing the fusion of the scores, by stacking the samples vertically in order for a "dataset" 
    of size (2, N). Works similarly as the calibrate_scores() function.'''

    csf_1_s = vrow(csf_1_scores)
    csf_2_s = vrow(csf_2_scores)

    scores_D = np.vstack([csf_1_s, csf_2_s])
    if w is None and b is None:
        logRegObj = logRegClass(scores_D, L, 0, eff_prior)
        logreg_scores, w, b = logRegObj.logreg_scores(scores_D, calibrate=True)
        fused_scores = logreg_scores - np.log(eff_prior / (1 - eff_prior))
        return fused_scores, w, b
    else:
        fused_scores = np.dot(w.T, scores_D) + b
        return fused_scores.ravel()