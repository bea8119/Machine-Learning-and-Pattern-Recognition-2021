from utils import vrow, vcol
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

class SVM_class:
    def __init__(self, DTR, LTR, K, kern=False, Poly_RBF=True, c=None, d=None, gamma=None):
        self.DTR = extended_D(DTR, K) if kern else DTR
        self.kern = kern
        self.z = 2 * LTR - 1
        self.K = K
        self.H = H_kern_matrix(DTR, self.z, Poly_RBF, K, c, d, gamma) if kern else H_matrix(self.D_ext, self.z)

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

    def SVM_llrs(self, DTE, bounds_list):
        alpha_0 = np.zeros(self.DTR.shape[0])
        (alpha_opt, L_dual_opt, d) = scipy.optimize.fmin_l_bfgs_b(
            self.kernel_SVM_obj if self.kern else self.linear_SVM_obj,
            alpha_0,
            bounds=bounds_list,
            factr=1.0
        )
        w_star = vcol(np.sum(vrow(alpha_opt) * vrow(self.z) * self.DTR, axis=1))
        scores = np.dot(w_star.T, DTE)# Acts as llrs for min DCF but actually not probabilistic so need to calibrate 
        return scores

