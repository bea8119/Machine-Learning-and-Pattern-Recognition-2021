from utils import vcol, vrow, split_dataset
from feature_utils import covMatrix, SW_compute
import numpy as np
import scipy.special

def logpdf_GAU_ND(x, mu, C):
    mu = vcol(mu)
    M = x.shape[0]
    N = x.shape[1]
    C_inv = np.linalg.inv(C)
    det_logC = np.linalg.slogdet(C)[1]

    Y = np.zeros(N)
    for i in range(N):
        x_mu = vcol(x[:, i]) - mu
        mat_product = (0.5 * (np.dot(np.dot(x_mu.T, C_inv), x_mu)))
        Y[i] = - (M * 0.5) * np.log(2 * np.pi) - (0.5 * det_logC) - mat_product
    return Y

def loglikelihood(XND, m_ML, C_ML):
    N = XND.shape[1]
    return np.sum([logpdf_GAU_ND(vcol(XND[:, i]), m_ML, C_ML) for i in range(N)])

def classifierSetup(D, L, k, idxTrain, idxTest, tied=False): 
    # Gaussian Classifier
    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, idxTrain, idxTest)

    mu_arr = [vcol(DTR[:, LTR == i].mean(axis=1)) for i in range(k)]

    if tied == False:
        C_arr = [covMatrix(DTR[:, LTR == i]) for i in range(k)]
    else:
        C_tied = SW_compute(DTR, LTR, k)
        C_arr = [C_tied for i in range(k)]
    return (DTE, LTE), mu_arr, C_arr 

def testLabelPredAccuracy(PredictedL, LTE, classifierName, show):
    CorrectPred = (PredictedL == LTE) # Boolean array of predictions
    CorrectCount = sum(CorrectPred)
    accuracy = CorrectCount / LTE.shape[0]
    error_rate = 1 - accuracy
    if (show):
        # print('Accuracy of' + classifierName + ' Gaussian classifier: ' + str(round(accuracy * 100, 2)) + ' %')
        print(classifierName + 'Gaussian classifier error rate: ' + str(round(error_rate * 100, 1)) + ' %')
    return LTE.size - CorrectCount # Error count

def gaussianCSF(DTE, LTE, k, mu_arr, C_arr, priorP, CSF_name, show):
    '''
    Returns the Predicted labels in a 1-D ndarray and the error count
    '''
     # now focus on the TEST samples
    N_test = DTE.shape[1]
    S = np.zeros((k, N_test))
    for i in range(k):
        # S will store the LOG densities
        S[i, :] = vrow(np.array([logpdf_GAU_ND(vcol(DTE[:, j]), mu_arr[i], C_arr[i]) for j in range(N_test)]))

    logSJoint = S + np.log(priorP)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0)) # compute for each column
    logSPost = logSJoint - logSMarginal
    SPost_afterLog = np.exp(logSPost)

    PredictedL = SPost_afterLog.argmax(0) # INDEX of max through axis 0 (per sample)

    err_count = testLabelPredAccuracy(PredictedL, LTE, CSF_name, show)
    
    return PredictedL, err_count

def gaussianCSF_wrapper(D, L, k, idxTrain, idxTest, priorP, show=True):
    (DTE, LTE), mu_arr, C_arr = classifierSetup(D, L, k, idxTrain, idxTest)
    return gaussianCSF(DTE, LTE, k, mu_arr, C_arr, priorP, "", show)

def naiveBayesGaussianCSF(D, L, k, idxTrain, idxTest, priorP, show=True):
    (DTE, LTE), mu_arr, C_arr = classifierSetup(D, L, k, idxTrain, idxTest)
    C_naive_arr = [C_arr[i] * np.identity(C_arr[i].shape[0]) for i in range(k)] # element by element mult.
    return gaussianCSF(DTE, LTE, k, mu_arr, C_naive_arr, priorP, "Naive Bayes ", show)

def tiedCovarianceGaussianCSF(D, L, k, idxTrain, idxTest, priorP, show=True):
    (DTE, LTE), mu_arr, C_arr = classifierSetup(D, L, k, idxTrain, idxTest, tied=True)
    return gaussianCSF(DTE, LTE, k, mu_arr, C_arr, priorP, "Tied Covariance ", show)

def tiedNaiveBayesGaussianCSF(D, L, k, idxTrain, idxTest, priorP, show=True):
    (DTE, LTE), mu_arr, C_arr = classifierSetup(D, L, k, idxTrain, idxTest, tied=True)
    C_naive_arr = [C_arr[i] * np.identity(C_arr[i].shape[0]) for i in range(k)] # element by element mult.
    return gaussianCSF(DTE, LTE, k, mu_arr, C_naive_arr, priorP, "Tied Naive Bayes ", show)

def K_fold_MVG(D, L, k, priorP, K, classifiers, seed=0):
    print(f'{K}-Fold cross-validation error rates (MVG Classifiers)')
    nTest = int(D.shape[1] / K)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) # take a random order of indexes from 0 to N
    # idx = np.arange(D.shape[1]) # take a random order of indexes from 0 to N

    error_acc = np.zeros((len(classifiers)))

    for i in range(len(classifiers)):
        startTest = 0
        for j in range(K):
            idxTest = idx[startTest: (startTest + nTest)] # take a different fold for test
            # print(idxTest)
            idxTrain = np.setdiff1d(idx, idxTest) # take as training set the remaining folds
            # print(idxTrain)
            error_acc[i] += classifiers[i][0](D, L, k, idxTrain, idxTest, priorP, show=False)[1]
            startTest += nTest

    error_rates = error_acc / D.shape[1] * 100 # Each data has been used as validation set, so you divide D.shape[1]
    for i in range(len(classifiers)):
        print(f'Classifier {i+1} ({classifiers[i][1]}) error rate: {round(error_rates[i], 1)} %')