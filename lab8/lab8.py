import numpy as np
import sys
import scipy.special
import matplotlib.pyplot as plt

# sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from lab6.lab6 import compute_classPosteriorP
from lab5.lab5 import gaussianCSF_wrapper, naiveBayesGaussianCSF, tiedCovarianceGaussianCSF, tiedNaiveBayesGaussianClassifier
from lab5.lab5 import split_db_2to1, split_dataset
from lab2.load_plot import load
from utility.vrow_vcol import vcol, vrow


def compute_confusion_matrix(predL, trueL, K):
	conf_matrix = np.zeros((K, K))
	for pCls, tCls in zip(predL, trueL):
		conf_matrix[pCls, tCls] += 1

	return conf_matrix


def compute_optBayes_decisions(llrs=None, pi_1=None, C_fn=None, C_fp=None, given_threshold=None):
	if (given_threshold is None):
		threshold = - np.log((pi_1 * C_fn) / ((1 - pi_1) * C_fp))
	else:
		threshold = given_threshold
	return np.int32(llrs > threshold)  # Genius method by Francesco
	# return np.array([(1 if llr > threshold else 0) for llr in llrs.ravel()])


def DCF_binary(confusion_m, pi_1, C_fn, C_fp):
	FNR = confusion_m[0, 1] / (confusion_m[0, 1] + confusion_m[1, 1])
	FPR = confusion_m[1, 0] / (confusion_m[0, 0] + confusion_m[1, 0])
	return pi_1 * C_fn * FNR + (1 - pi_1) * C_fp * FPR


def FNR_FPR(confusion_m):
	'''Returns FNR, FPR given a confusion matrix (binary)'''
	return confusion_m[0, 1] / (confusion_m[0, 1] + confusion_m[1, 1]), confusion_m[1, 0] / (confusion_m[0, 0] + confusion_m[1, 0])


def ROC_TPR_vs_FPR(llrs, trueL):
	thresholds = np.array(llrs)
	thresholds.sort()
	thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
	TPR_arr = np.zeros(thresholds.shape[0])
	FPR_arr = np.zeros(thresholds.shape[0])
	for idx, t in enumerate(thresholds):
		conf_m_temp = compute_confusion_matrix(compute_optBayes_decisions(llrs, given_threshold=t), trueL, 2)
		FNR_temp, FPR_temp = FNR_FPR(conf_m_temp)
		TPR_arr[idx] = 1 - FNR_temp
		FPR_arr[idx] = FPR_temp
	return TPR_arr, FPR_arr

def DCF_unnormalized_normalized_min(llrs, trueL, triplet):
	# Un-normalized
	dcf_u = DCF_binary(compute_confusion_matrix(compute_optBayes_decisions(llrs, *triplet), trueL, 2), *triplet)
	# Bayesian risk (with dummy system)
	B_dummy = min(triplet[0] * triplet[1], (1 - triplet[0]) * triplet[2])
	# Normalized Detection Cost Function (wrt. to dummy system)
	dcf_norm = dcf_u / B_dummy

	# Minimum DCF (with score calibration)
	# Create a new object (ndarray), otherwise in-place sorting will mess up the following computations of the confusion matrix
	thresholds = np.array(llrs)
	thresholds.sort()
	thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])

	dcf_min = np.inf

	for threshold in thresholds:
		dcf_temp = DCF_binary(compute_confusion_matrix(compute_optBayes_decisions(llrs, *triplet, threshold), trueL, 2), *triplet)
		dcf_temp_norm = dcf_temp / B_dummy
		if dcf_temp_norm < dcf_min:
			dcf_min = dcf_temp_norm 
	return (dcf_u, dcf_norm, dcf_min)

def DCF_vs_priorLogOdds(effPriorLogOdds, llrs, trueL):
	'''Returns normalized and min DCF for a given set of effective prior log-odds'''
	dcf_arr = np.zeros(effPriorLogOdds.shape[0])
	dcfmin_arr = np.zeros(effPriorLogOdds.shape[0])
	for idx, p in enumerate(effPriorLogOdds):
		eff_pi = 1 / (1 + np.exp(-p))
		dcfs = DCF_unnormalized_normalized_min(llrs, trueL, (eff_pi, 1, 1))
		dcf_arr[idx] = dcfs[1]
		dcfmin_arr[idx] = dcfs[2]
	return dcf_arr, dcfmin_arr

def main():
	# --------------- MVG clasiffiers ---------------

	# IRIS dataset

	# D, L = load('../datasets/iris.csv')
	# priorP = vcol(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
	# K = len(np.unique(L)) # Number of distinct classes

	# idxTrain, idxTest = split_db_2to1(D)
	# LTE = L[idxTest]

	# gaussianL = gaussianCSF_wrapper(D, L, K, idxTrain, idxTest, priorP, log=False, show=False)
	# # print(compute_confusion_matrix(gaussianL, LTE, K))

	# naiveBayesGL = naiveBayesGaussianCSF(D, L, K, idxTrain, idxTest, priorP, log=False, show=False)
	# # print(compute_confusion_matrix(naiveBayesGL, LTE, K))

	# tiedCovGL = tiedCovarianceGaussianCSF(D, L, K, idxTrain, idxTest, priorP, log=False, show=False)
	# # print(compute_confusion_matrix(tiedCovGL, LTE, K))

	# tiedNaiveBayesGL = tiedNaiveBayesGaussianClassifier(D, L, K, idxTrain, idxTest, priorP, log=False, show=False)
	# # print(compute_confusion_matrix(tiedNaiveBayesGL, LTE, K))

	# ----------------- Divina Commedia -------------

	divinaComediaLabels = np.load('data/commedia_labels.npy')
	K = len(np.unique(divinaComediaLabels))

	divinaCommediaLL = np.load('data/commedia_ll.npy')

	priorP_log = vcol(np.log(np.array([1./3., 1./3., 1./3.])))
	scoreMatrix = compute_classPosteriorP(divinaCommediaLL, priorP_log)
	predLCommedia = np.argmax(scoreMatrix, axis=0)

	print('Divina Commedia confusion matrix:\n', compute_confusion_matrix(predLCommedia, divinaComediaLabels, K))

	# ------------- Binary task: optimal decisions ------------------

	# Inferno - Paradiso (Divina Commedia)
	llr_infpar = np.load('data/commedia_llr_infpar.npy')
	labels_infpar = np.load('data/commedia_labels_infpar.npy')

	prior_1 = 0.5
	C_fn = 1
	C_fp = 1
	# cost_matrix = np.array([[0, C_fn], [C_fp, 0]])
	predLCommedia_optimalD = compute_optBayes_decisions(llr_infpar, prior_1, C_fn, C_fp)
	conf_matrix = compute_confusion_matrix(predLCommedia_optimalD, labels_infpar, 2)

	triplets = [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]
	print(f'Optimal Bayes decisions (binary) confusion matrix with\npi_1: {prior_1}, C_fn: {C_fn}, C_fp: {C_fp}:\n', conf_matrix)

	for triplet in triplets:
		(dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min(llr_infpar, labels_infpar, triplet)
		print('pi_1: {}, C_fn: {}, C_fp: {}:\tDCF_u:{}\tDCF_norm: {}\tDCF_min: {}'.format(*triplet, dcf_u, dcf_norm, dcf_min))

	# ---------------------- ROC curves ----------------------

	# Outside of the for loop because this does not depend on the the application (triplet) but just on the threshold
	TPR_arr, FPR_arr = ROC_TPR_vs_FPR(llr_infpar, labels_infpar)

	plt.figure('ROC')
	plt.plot(FPR_arr, TPR_arr)
	plt.grid()
	plt.xlabel('FPR')
	plt.ylabel('TPR')

	# ------------------- Bayes error plots -------------------

	effPriorLogOdds = np.linspace(-3, 3, 21)
	dcf_arr, dcfmin_arr = DCF_vs_priorLogOdds(effPriorLogOdds, llr_infpar, labels_infpar)
	
	# Loading eps = 1 dataset
	llr_infpar_eps1 = np.load('data/commedia_llr_infpar_eps1.npy')
	labels_infpar_eps1 = np.load('data/commedia_labels_infpar_eps1.npy')

	dcf_arr_eps1, dcfmin_arr_eps1 = DCF_vs_priorLogOdds(effPriorLogOdds, llr_infpar_eps1, labels_infpar_eps1)
	
	plt.figure('Bayes error plots')
	plt.plot(effPriorLogOdds, dcf_arr, label='DCF (eps = 0.001)', color='r')
	plt.plot(effPriorLogOdds, dcfmin_arr, label='min DCF (eps = 0.001)', color='b')
	plt.plot(effPriorLogOdds, dcf_arr_eps1, label='DCF (eps = 1)', color='y')
	plt.plot(effPriorLogOdds, dcfmin_arr_eps1, label='min DCF (eps = 1)', color='c')
	plt.xlim([-3, 3])
	plt.xlabel('Prior log-odds')
	plt.ylim([0, 1.1])
	plt.ylabel('DCF value')
	plt.grid()
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
