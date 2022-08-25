from utils import vcol, split_dataset
from feature_utils import PCA_givenM
from DCF import DCF_unnormalized_normalized_min_binary
import numpy as np

def lambdaTuning()

def computeAccuracy_logreg_binary(scoreArray, TrueL):
    PredictedL = np.array([(1 if score > 0 else 0) for score in scoreArray.ravel()])
    NCorrect = (PredictedL.ravel() == TrueL.ravel()).sum() # Will count as 1 the "True"
    NTotal = TrueL.size
    return float(NCorrect) / float(NTotal)

class logRegClass:
    def __init__(self, DTR, LTR, l, priorT, K=2):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.priorT = priorT
        self.K = K
    def logreg_obj_binary(self, v):
        w, b = vcol(v[0:-1]), v[-1]
        s0 = np.dot(w.T, self.DTR[:, self.LTR == 0]) + b
        s1 = np.dot(w.T, self.DTR[:, self.LTR == 1]) + b
        z = 2.0 * self.LTR - 1 # Encoding to +1 and -1

        # Cross-entropy
        mean_term1 = np.logaddexp(0, -z * s1).mean()
        mean_term0 = np.logaddexp(0, -z * s0).mean()
        crossEntropy = self.priorT * mean_term1 + (1 - self.priorT) * mean_term0
        return 0.5 * self.l * np.linalg.norm(w)**2 + crossEntropy

def K_fold_LogReg(D, L, k, K, classifiers, app_triplet, PCA_m=None, seed=0):
    if PCA_m is not None:
        msg = f' with PCA m={PCA_m}'
    else: 
        msg = ' (no PCA)'
    print(f'{K}-Fold cross-validation (MVG Classifiers){msg}')
    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) 

    for i in range(len(classifiers)):
        startTest = 0
        # For DCF computation
        llrs = np.array([])
        for j in range(K):
            idxTest = idx[startTest: (startTest + nTest)]
            idxTrain = np.setdiff1d(idx, idxTest)
            if PCA_m is not None:
                DTR_PCA_fold = split_dataset(D, L, idxTrain, idxTest)[0][0]
                PCA_P = PCA_givenM(DTR_PCA_fold, PCA_m)
                D_PCA = np.dot(PCA_P.T, D)
                log_likelihoods = classifiers[i][0](D_PCA, L, k, idxTrain, idxTest, app_triplet, show=False)
            else:
                log_likelihoods = classifiers[i][0](D, L, k, idxTrain, idxTest, app_triplet, show=False)

            llr = log_likelihoods[1, :] - log_likelihoods[0, :] # log-likelihood ratio
            llrs = np.concatenate((llrs, llr))
            startTest += nTest
        
        # DCF computation (compute)
        trueL_ordered = L[idx] # idx was computed randomly before
        (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(llrs, trueL_ordered, app_triplet)
        print(f'\t{classifiers[i][1]} classifier -> min DCF: {round(dcf_min, 3)}')