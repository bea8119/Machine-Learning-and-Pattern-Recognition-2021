import numpy as np
import sys
import json
import scipy.special, scipy.optimize
import matplotlib.pyplot as plt

# sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from lab7.lab7 import load_iris_binary, computeAccuracy_logreg_binary
from lab6.lab6 import compute_classPosteriorP, compute_logLikelihoods
from lab5.lab5 import split_db_2to1, split_dataset
from lab4.lab4 import logpdf_GAU_ND
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

def M_step(gamma, X):
    '''
    Receives responsibilities (gamma) which is a (MxN) matrix representing a posterior probabilities of the GMM of each sample.\n
    Returns new estimated parameters in a list of tuples (means, cov_matrices, weights).
    '''
    new_gmm = []
    tot_Z = 0

    Z_list = []
    mean_list = []
    c_matrix_list = []

    for gamma_g in gamma: # Iterate over each row
        Z = np.sum(gamma_g)
        Z_list.append(Z)

        tot_Z += Z

        F = vcol(np.dot(vrow(gamma_g), X.T))
        S = np.dot(vrow(gamma_g) * X, X.T) # Exploits broadcasting
        # S = np.dot(X, (vrow(gamma_g) * X).T)

        mu_next = F / Z
        # print(mu_next)
        cov_m_next = (S / Z) - np.dot(mu_next, mu_next.T)
        # print(cov_m_next.shape)

        mean_list.append(mu_next)
        c_matrix_list.append(cov_m_next)


    for i in range(gamma.shape[0]):
        weight_next = Z_list[i] / tot_Z
        new_gmm.append((weight_next, mean_list[i], c_matrix_list[i]))


    return new_gmm


def main():

    # ------------------ Gaussian Mixture models ----------------

    GMM_data = np.load('data/GMM_data_4D.npy')
    ref_GMM = load_gmm('data/GMM_4D_3G_init.json')

    # lls = logpdf_GMM(GMM_data, ref_GMM)[0]
    # sol_GMM_ll = np.load('data/GMM_4D_3G_init_ll.npy')
    # print(np.abs(sol_GMM_ll - lls).max()) # Test passed

    # ------------------- EM-algorithm ----------------------

    i = 0
    delta = 1e-6
    X = GMM_data
    N = X.shape[1]

    avg_log_ll = 0 # Initialization
    new_gmm = ref_GMM # Initiailization

    while True:
        prev_avg_ll = avg_log_ll
        
        gamma, log_ll = E_step(X, new_gmm)
        avg_log_ll = log_ll / N

        if abs((avg_log_ll - prev_avg_ll)) < delta:
            print('fua')
            break
        
        new_gmm = M_step(gamma, X)
        print('Iteration', i, ': ', avg_log_ll)
        i += 1




if __name__ == '__main__':
    main()
