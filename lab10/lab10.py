import numpy as np
import sys
import json
import scipy.special, scipy.optimize
import matplotlib.pyplot as plt

sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
# sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from lab7.lab7 import load_iris_binary, computeAccuracy_logreg_binary
from lab6.lab6 import compute_accuracy, compute_classPosteriorP, compute_logLikelihoods
from lab5.lab5 import split_db_2to1, split_dataset
from lab4.lab4 import logpdf_GAU_ND
from lab3.lab3 import datasetCovarianceM
from lab2.load_plot import load
from utility.vrow_vcol import vcol, vrow

def logpdf_GMM(X, gmm):
    '''
    Compute Marginal log-density of a GMM, X is a matrix of samples of shape (D, N) (standard D matrix).\n
    gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]. Returns SMarginal and SJoint
    '''
    M = len(gmm)
    N = X.shape[1]

    S = np.zeros((M, N))

    for (g, (weight, mean, C_matrix)) in enumerate(gmm):
        S[g, :] = logpdf_GAU_ND(X, mean, C_matrix)
        S[g, :] += np.log(weight) # Adding the log prior

    return scipy.special.logsumexp(S, axis=0), S

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

def compute_clusterPosteriorP(SJoint):
    '''
    Receives SJ.
    Similar to compute_classPosteriorP from lab6, but this assumes the prior term has already been added.\n
    Returns responsibility matrix.
    '''
    logSMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0)) # compute for each column
    logSPost = SJoint - logSMarginal
    SPost = np.exp(logSPost)

    return SPost

def compute_logLikelihoods_GMM(lls_array):
    return lls_array.sum()

def E_step(X, gmm):
    '''Computes responsibilities (gamma) matrix. Returns also log-likelihood of the previous estimate'''
    # Re-use logpdf_GMM
    SM, SJ = logpdf_GMM(X, gmm)
    ll = compute_logLikelihoods_GMM(SM)
    # print(S.shape)

    gamma = compute_clusterPosteriorP(SJ)
    # print(gamma.shape)
    return gamma, ll

def M_step(gamma, X, psi, diagCov=False, tiedCov=False):
    '''
    Receives responsibilities (gamma) which is a (MxN) matrix representing a posterior probabilities of the GMM of each sample.\n
    Returns new estimated parameters in a list of tuples (means, cov_matrices, weights).
    Adjusted to avoid degenerate solutions (during generation of Covariance matrices)
    '''
    new_gmm = []
    tot_Z = 0

    Z_list = []
    mean_list = []
    c_matrix_list = []
    tied_cov_matrix = np.zeros((X.shape[0], X.shape[0]))

    for gamma_g in gamma: # Iterate over each row
        Z = np.sum(gamma_g)
        Z_list.append(Z)

        tot_Z += Z

        F = vcol(np.dot(vrow(gamma_g), X.T))
        S = np.dot(vrow(gamma_g) * X, X.T) # Exploits broadcasting

        mu_next = F / Z
        mean_list.append(mu_next)

        cov_m_next = (S / Z) - np.dot(mu_next, mu_next.T)

        # Diagonal or Tied Covariance update
        if diagCov == True:
            cov_m_next = cov_m_next * np.eye(cov_m_next.shape[0])
        elif tiedCov == True:
            tied_cov_matrix +=  Z * cov_m_next
        
        c_matrix_list.append(cov_m_next)
        
    if tiedCov == True:
        tied_cov_matrix = tied_cov_matrix / X.shape[1]

    for i in range(gamma.shape[0]):
        weight_next = Z_list[i] / tot_Z
        if tiedCov == True:
            c_matrix_list[i] = tied_cov_matrix

        # Degenerate adjustment (constraining eigenvalues of cov. matrix)
        U, s, _ = np.linalg.svd(c_matrix_list[i])
        s[s < psi] = psi
        c_matrix_list[i] = np.dot(U, vcol(s) * U.T)

        new_gmm.append((weight_next, mean_list[i], c_matrix_list[i]))


    return new_gmm

def EM_GMM(X, init_GMM, delta, psi, diagCov=False, tiedCov=False, iprint=False):
    '''
    Performs EM algorithm for estimating better GMM model parameters (weight, mean, cov).\n
    Returns optimized new GMM models
    '''
    i = 0
    N = X.shape[1]
    new_gmm = init_GMM # Initiailization
    avg_log_ll = 0 # Initialization


    while True:
        prev_avg_ll = avg_log_ll
        
        gamma, log_ll = E_step(X, new_gmm)
        avg_log_ll = log_ll / N

        if abs((avg_log_ll - prev_avg_ll)) < delta:
            return new_gmm
        
        new_gmm = M_step(gamma, X, psi, diagCov, tiedCov)

        if iprint == True:
            print('Iteration', i, ': ', avg_log_ll)
            i += 1

def LBG_GMM(X, init_GMM, delta, alpha, psi, n_splits, iprint=False, diagCov=False, tiedCov=False):
    '''
    Returns optimized GMM components (and model parameters) using the LBG algorithm.
    diagCov and tiedCov flags should be passed for choosing the 2 variants. If not provided, the algorithm will
    run in the "Full Covariance" manner.
    '''
    curr_GMM = init_GMM
    for i in range(n_splits):
        new_GMM_split = []
        for gmm in curr_GMM:
            # Split
            new_w = gmm[0] / 2

            U, s, Vh = np.linalg.svd(gmm[2])
            d = U[:, 0:1] * s[0]**0.5 * alpha

            new_mu1 = gmm[1] + d
            new_mu2 = gmm[1] - d

            new_GMM_split.append((new_w, new_mu1, gmm[2]))
            new_GMM_split.append((new_w, new_mu2, gmm[2]))

        # Train (AFTER finishing splitting)
        new_GMM = EM_GMM(X, new_GMM_split, delta, psi, iprint=iprint, diagCov=diagCov, tiedCov=tiedCov)
        curr_GMM = new_GMM
    return curr_GMM

def GMM_classifier(DTR, LTR, DTE, LTE, K, delta, alpha, psi, n_splits, priorP, diagCov=False, tiedCov=False, iprint=False):
    # Train the model for each class
    if diagCov:
        tag = ' (Diagonal Cov)'
    elif tiedCov:
        tag = ' (Tied Cov)'
    else: 
        tag = ''
    SM_cluster = np.zeros((K, DTE.shape[1]))

    for i in range(K):
        cls_data = DTR[:, LTR == i] # Subset of samples of DTR that belong to the same class

        # Compute ML parameters as starting point for LBG
        mu_ML = vcol(cls_data.mean(axis=1))
        C_ML = datasetCovarianceM(cls_data)
        U, s, _ = np.linalg.svd(C_ML)
        s[s < psi] = psi
        C_ML_adjusted = np.dot(U, vcol(s) * U.T)

        GMM_1_ML = [(1.0, mu_ML, C_ML_adjusted)]

        gmm_LBG = LBG_GMM(cls_data, GMM_1_ML, delta, alpha, psi, n_splits, iprint=iprint, diagCov=diagCov, tiedCov=tiedCov)
        # We have optimal GMM for this class
        SM_cluster[i, :], _ = logpdf_GMM(DTE, gmm_LBG) # AKA the class joint density of a "MVG"
    
    posteriorP_TE = compute_classPosteriorP(SM_cluster, np.log(priorP))
    err_rate = 1 - compute_accuracy(posteriorP_TE, LTE)

    print(f'Error rate GMM classifier{tag} with {2**n_splits} components:', str(round(err_rate * 100, 1)), '%')

def main():

    # ------------------ Gaussian Mixture models ----------------

    GMM_data = np.load('data/GMM_data_4D.npy')
    ref_GMM = load_gmm('data/GMM_4D_3G_init.json')

    lls = logpdf_GMM(GMM_data, ref_GMM)[0]
    sol_GMM_ll = np.load('data/GMM_4D_3G_init_ll.npy')
    # print(np.abs(sol_GMM_ll - lls).max()) # Test passed

    # ------------------- EM-algorithm ----------------------

    delta = 1e-6
    psi = 0 # For constraining eigenvalues of cov matrix
    # opt_EM_gmm = EM_GMM(GMM_data, ref_GMM, delta, psi)

    # ------------------- LBG algorithm ---------------------

    # Maximum likelihood parameters
    mu_ML = vcol(GMM_data.mean(axis=1))
    C_ML = datasetCovarianceM(GMM_data)

    
    U, s, _ = np.linalg.svd(C_ML)
    s[s < psi] = psi
    C_ML_adjusted = np.dot(U, vcol(s) * U.T)
    # IMPORTANT: CHECK that this transformation is important in the ML solution with other dataset, because in this ML solution,
    # the singular value vector doesn't contain any eigenvalue lower than 0 so the initial transformation seems to do nothing

    # print(np.abs(C_ML_adjusted - C_ML).max())

    GMM_1 = [(1.0, mu_ML, C_ML_adjusted)]
    alpha = 0.1
    
    opt_EM_GMM = EM_GMM(GMM_data, GMM_1, delta, psi, iprint=True)
    print('end of step 0 EM')
    opt_LBG_GMM = LBG_GMM(GMM_data, GMM_1, delta, alpha, psi, 2, iprint=True)

    # ----------------- GMM for classification ---------------------

    D, L = load('../datasets/iris.csv')
    K = len(np.unique(L))
    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, *split_db_2to1(D))
    delta = 1e-6
    alpha = 0.1
    psi = 0.01
    priorP = vcol(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
    n_splits = 4 # Number of components will be equal to 2^n_splits

    for i in range(n_splits + 1):
        GMM_classifier(DTR, LTR, DTE, LTE, K, delta, alpha, psi, i, priorP)
        GMM_classifier(DTR, LTR, DTE, LTE, K, delta, alpha, psi, i, priorP, diagCov=True)
        GMM_classifier(DTR, LTR, DTE, LTE, K, delta, alpha, psi, i, priorP, tiedCov=True)
        print('-----------------------------------')
    


if __name__ == '__main__':
    main()
