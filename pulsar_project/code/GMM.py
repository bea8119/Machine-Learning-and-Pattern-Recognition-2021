from MVG import logpdf_GAU_ND
from utils import vrow, vcol, split_dataset
from feature_utils import covMatrix, Z_normalization, PCA_givenM
from DCF import DCF_unnormalized_normalized_min_binary
from LogReg import calibrate_scores
import numpy as np
import scipy.special

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
        S[g, :] += np.log(weight) # Adding the log prior weight

    return scipy.special.logsumexp(S, axis=0), S

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

        # Diagonal and/or Tied Covariance update
        if diagCov == True:
            cov_m_next = cov_m_next * np.eye(cov_m_next.shape[0])
        if tiedCov == True:
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

def GMM_wrapper(D, L, k, idxTrain, idxTest, delta, alpha, psi, n_splits, tied=False, diag=False, triplet=None, single_fold=True, show=True, iprint=False, calibrate=False):
    GMM_type = ''
    if tied:
        GMM_type += 'Tied '
    if diag:
        GMM_type += 'Diag '
    if not diag and not tied:
        GMM_type += 'Full '
    GMM_type += 'Covariance '

    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, idxTrain, idxTest)

    # Apply Z-normalization on the training set (of the current fold), apply the same transformation on the test set
    DTR, mean, std = Z_normalization(DTR)
    DTE = Z_normalization(DTE, mean, std)

    scores = GMM_classifier(DTR, LTR, DTE, k, delta, alpha, psi, n_splits, tied, diag, iprint)

    if calibrate:
        scores, w, b = calibrate_scores(scores, LTE, 0.5)

    if single_fold:
        return testDCF_GMM(LTE, GMM_type, n_splits, scores, triplet)
    return scores

def GMM_classifier(DTR, LTR, DTE, k, delta, alpha, psi, n_splits, tiedCov=False, diagCov=False, iprint=False):
    # Train the model for each class
    PP_cluster = np.zeros((k, DTE.shape[1]))

    for i in range(k):
        cls_data = DTR[:, LTR == i] # Subset of samples of DTR that belong to the same class
        # Compute ML parameters as starting point for LBG
        mu_ML = vcol(cls_data.mean(axis=1))
        C_ML = covMatrix(cls_data)
        U, s, _ = np.linalg.svd(C_ML)
        s[s < psi] = psi
        C_ML_adjusted = np.dot(U, vcol(s) * U.T)

        GMM_1_ML = [(1.0, mu_ML, C_ML_adjusted)]

        gmm_LBG = LBG_GMM(cls_data, GMM_1_ML, delta, alpha, psi, n_splits, iprint=iprint, diagCov=diagCov, tiedCov=tiedCov)
        # We have optimal GMM for this class
        PP_cluster[i, :], _ = logpdf_GMM(DTE, gmm_LBG) # AKA the class-conditional log density of a "MVG" (S matrix)

    scores = PP_cluster[1, :] - PP_cluster[0, :]
    return scores

def testDCF_GMM(LTE, classifierName, n_splits, llrs, triplet, show=True):
    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(llrs, LTE, triplet)
    if show:
        print(f'\t{classifierName}GMM classifier ({2**n_splits} components)-> min DCF: {round(dcf_min, 3)}    act DCF: {round(dcf_norm, 3)}')
    return dcf_min

def K_fold_GMM(D, L, k, K, delta, alpha, psi, n_splits, tied, diag, app_triplet, PCA_m=None, show=True, seed=0, printStatus=False, calibrate=False):
    if show:
        GMM_type = ''
        if tied:
            GMM_type += 'Tied '
        if diag:
            GMM_type += 'Diag '
        if not diag and not tied:
            GMM_type += 'Full '
        GMM_type += 'Covariance '

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
            scores = GMM_wrapper(D_PCA, L, k, idxTrain, idxTest, delta, alpha, psi, n_splits, tied, diag, app_triplet, False, show)
        else:
            scores = GMM_wrapper(D, L, k, idxTrain, idxTest, delta, alpha, psi, n_splits, tied, diag, app_triplet, False, show)

        scores_all = np.concatenate((scores_all, scores))
        startTest += nTest
        
    # DCF computation (compute)
    trueL_ordered = L[idx] # idx was computed randomly before

    if calibrate:
        scores_all, w, b = calibrate_scores(scores_all, trueL_ordered, 0.5)

    if printStatus:
        print('calculating minDCF...')

    (dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(scores_all, trueL_ordered, app_triplet)
    if show:
        print('\t{}GMM (n_components = {}) -> min DCF: {}    act DCF: {}'.format(GMM_type, 2**n_splits, round(dcf_min, 3), round(dcf_norm, 3)))
    return dcf_min