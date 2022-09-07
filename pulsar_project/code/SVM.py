from utils import vrow, vcol, split_dataset
from feature_utils import Z_normalization, PCA_givenM
from LogReg import calibrate_scores
from DCF import DCF_unnormalized_normalized_min_binary
import numpy as np
import scipy.optimize

def extended_D(sample_vector, K):
    # K is not the number of classes, it is the factor of extension of the dataset
    return np.vstack([sample_vector, np.full((1, sample_vector.shape[1]), K)])

def H_matrix(extended_D, encodedL):
    G_hat = np.dot(extended_D.T, extended_D)
    # Broadcast a row and a column vector of the labels for multiplying z_i and z_j to G_hat
    return G_hat * vrow(encodedL) * vcol(encodedL)

def compute_duality_gap(w_star, C, D_ext, encodedL, L_Dual_opt):
    scores = np.dot(w_star.T, D_ext)
    loss_term = np.sum(np.maximum(np.zeros(D_ext.shape[1]), 1 - encodedL * scores))
    return (0.5 * np.linalg.norm(w_star)**2 + C * loss_term) + L_Dual_opt

def H_kern_matrix(D, encodedL, Poly_RBF, K, c, d, gamma):
    if Poly_RBF:     
        k_f_matrix = kernel_func_Poly(D, K, c, d)
    else:
        k_f_matrix = kernel_func_RBF(D, K, gamma)
    return k_f_matrix * vrow(encodedL) * vcol(encodedL)

def kernel_func_Poly(D, K, c, d):
    xi = K**2
    mat_prod = np.dot(D.T, D)
    return (mat_prod + c)**d + xi

def kernel_f_Poly_x1_x2(x1, x2, K, c, d, gamma):
    xi = K**2 
    mat_prod = np.dot(x1.T, x2)
    return (mat_prod + c)**d + xi

def kernel_func_RBF(D, K, gamma):
    xi = K**2
    n = D.shape[1]
    diff_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff_mat[i, j] = (np.linalg.norm(D[:, i] - D[:, j]))**2
    return np.exp(- gamma * diff_mat) + xi

def kernel_f_RBF_x1_x2(x1, x2, K, c, d, gamma):
    xi = K**2
    norm_term = np.linalg.norm(x1 - x2)
    return np.exp(- gamma * norm_term**2) + xi

def kernel_SVM_scores(DTR, DTE, alpha_star, encodedLTR, kern_f, K, c, d, gamma):
    n_te = DTE.shape[1]
    n_tr = DTR.shape[1]
    scores = np.zeros(n_te)
    for i in range(n_te):
        for j in range(n_tr):
            if (alpha_star[j] <= 0):
                continue
            scores[i] += alpha_star[j] * encodedLTR[j] * kern_f(DTR[:, j], DTE[:, i], K, c, d, gamma)
    return scores

def SVM_wrapper(D, L, K_svm, C, priorT_b, idxTrain, idxTest, triplet, c=None, d=None, gamma=None, single_fold=True, show=True, kern=False, Poly_RBF=True, calibrate=False):
    
    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, idxTrain, idxTest)

    # Apply Z-normalization on the training set (of the current fold), apply the same transformation on the test set
    DTR, mean, std = Z_normalization(DTR)
    DTE = Z_normalization(DTE, mean, std)
    if not kern:
        DTR = extended_D(DTR, K_svm)
        DTE = extended_D(DTE, K_svm)
    SVM_obj = SVM_class(DTR, LTR, K_svm, kern, Poly_RBF, c, d, gamma)

    if priorT_b is None:
        bounds_list = [(0, C) for i in range(DTR.shape[1])]
    else:
        priorT_emp = (LTR == 1).sum() / DTR.shape[1]
        priorF_emp = (LTR == 0).sum() / DTR.shape[1]
        C_T = C * priorT_b / priorT_emp
        C_F = C * (1 - priorT_b) / priorF_emp
        bounds_list = [(0, C_T if LTR[i] == 1 else C_F) for i in range(DTR.shape[1])]

    scores = SVM_obj.SVM_scores(DTE, kern, Poly_RBF, bounds_list, c, d, gamma)

    if calibrate:
        scores, w, b = calibrate_scores(scores, LTE, 0.5)

    if single_fold:
        return DCF_SVM(LTE, K_svm, C, scores, triplet, show, kern, Poly_RBF, priorT_b)
    return scores

def DCF_SVM(LTE, K_svm, C, scores, triplet, show=True, kern=False, Poly_RBF=True, priorT_b=None):
    '''Returns DCF min for C tuning'''
    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(scores, LTE, triplet)
    if show:
        if kern:
            type_SVM = '{} Kernel'.format('Quadratic' if Poly_RBF else 'RBF')
        else:
            type_SVM = 'Linear'
        print('\t{} SVM (K = {}, C = {}, priorT = {}) -> min DCF: {}    act DCF: {}'.format(
            type_SVM,
            K_svm, C, priorT_b if priorT_b is not None else 'unb', round(dcf_min, 3), round(dcf_norm, 3)))
    return dcf_min

class SVM_class:
    def __init__(self, DTR, LTR, K, kern=False, Poly_RBF=True, c=None, d=None, gamma=None):
        self.DTR = DTR
        self.z = 2 * LTR - 1
        self.K = K
        self.H = H_kern_matrix(DTR, self.z, Poly_RBF, K, c, d, gamma) if kern else H_matrix(self.DTR, self.z)

    def linear_SVM_obj(self, alpha):
        # alpha instead of v (different than lab 7), which is received as a 1-D array, returns the gradient too
        alpha_v = vcol(alpha)
        return ((0.5 * np.dot(np.dot(alpha_v.T, self.H), alpha_v)) - np.sum(alpha_v),
                np.dot(self.H, alpha_v) - np.ones((alpha_v.shape[0], 1))
                )

        # np.dot(alpha_v.T, np.ones((alpha_v.shape[0], 1)) is the more inefficient way of doing it
    def kernel_SVM_obj(self, alpha):
        alpha_v = vcol(alpha)
        return ((0.5 * np.dot(np.dot(alpha_v.T, self.H), alpha_v)) - np.sum(alpha_v),
                np.dot(self.H, alpha_v) - np.ones((alpha_v.shape[0], 1))
                )

    def SVM_scores(self, DTE, kern, PolyRBF, bounds_list, c=None, d=None, gamma=None):
        alpha_0 = np.zeros(self.DTR.shape[1])
        (alpha_opt, L_dual_opt, data) = scipy.optimize.fmin_l_bfgs_b(
            self.kernel_SVM_obj if kern else self.linear_SVM_obj,
            alpha_0,
            bounds=bounds_list,
            factr=1.0,
            # maxiter=100,
        )
        if kern:
            # Kernel SVM
           return kernel_SVM_scores(self.DTR, DTE, alpha_opt, self.z,
            kernel_f_Poly_x1_x2 if PolyRBF else kernel_f_RBF_x1_x2,
            self.K, c, d, gamma)
        else:
            # Linear SVM
            w_star = vcol(np.sum(vrow(alpha_opt) * vrow(self.z) * self.DTR, axis=1))
            scores = np.dot(w_star.T, DTE) # Acts as llrs for min DCF but actually not probabilistic so need to calibrate 
            return scores.ravel() # Return as 1D array

def K_fold_SVM(D, L, K, K_svm, C, priorT_b, app_triplet, PCA_m=None, seed=0, show=True, kern=False, Poly_RBF=True, c=None, d=None, gamma=None, calibrate=False, printStatus=False):
    if show:
        if kern:
            type_SVM = '{} Kernel'.format('Quadratic' if Poly_RBF else 'RBF')
        else:
            type_SVM = 'Linear'

    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) 

    startTest = 0
    # For DCF computation
    scores_all = np.array([])
    for j in range(K):
        if printStatus:
            print('fold {} start...'.format(j + 1))
        idxTest = idx[startTest: (startTest + nTest)]
        idxTrain = np.setdiff1d(idx, idxTest)
        if PCA_m is not None:
            DTR_PCA_fold = split_dataset(D, L, idxTrain, idxTest)[0][0]
            PCA_P = PCA_givenM(DTR_PCA_fold, PCA_m)
            D_PCA = np.dot(PCA_P.T, D)
            scores = SVM_wrapper(D_PCA, L, K_svm, C, priorT_b, idxTrain, idxTest, app_triplet,
                c, d, gamma, single_fold=False, show=show, kern=kern, Poly_RBF=Poly_RBF, calibrate=calibrate)
        else:
            scores = SVM_wrapper(D, L, K_svm, C, priorT_b, idxTrain, idxTest, app_triplet,
                c, d, gamma, single_fold=False, show=show, kern=kern, Poly_RBF=Poly_RBF, calibrate=calibrate)

        scores_all = np.concatenate((scores_all, scores))
        startTest += nTest

    # DCF computation (compute)
    trueL_ordered = L[idx] # idx was computed randomly before

    if calibrate:
        scores_all, w, b = calibrate_scores(scores_all, trueL_ordered, 0.5)

    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(scores_all, trueL_ordered, app_triplet)
    if show:
        print('\t{} SVM (K = {}, C = {}, priorT = {}) -> min DCF: {}    act DCF: {}'.format(
            type_SVM, K_svm, C, priorT_b if priorT_b else 'unb', round(dcf_min, 3), round(dcf_norm, 3)))
    return dcf_min
