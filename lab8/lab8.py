import numpy as np
import sys
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


def compute_optBayes_decisions(llrs, pi_1=None, C_fn=None, C_fp=None, given_threshold=None):
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
	FNR_arr = np.zeros(thresholds.shape[0])
	TPR_arr = np.zeros(thresholds.shape[0])
	FPR_arr = np.zeros(thresholds.shape[0])
	for idx, t in enumerate(thresholds):
		conf_m_temp = compute_confusion_matrix(compute_optBayes_decisions(llrs, given_threshold=t), trueL, 2)
		FNR_temp, FPR_temp = FNR_FPR(conf_m_temp)
		FNR_arr[idx] = FNR_temp
		TPR_arr[idx] = 1 - FNR_temp
		FPR_arr[idx] = FPR_temp
	return TPR_arr, FPR_arr, FNR_arr

def DCF_unnormalized_normalized_min_binary(llrs, trueL, triplet):
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

	# Vectorize the functions
	# compute_opt_v = np.vectorize(compute_optBayes_decisions, otypes=[np.ndarray], excluded=[0, 1, 2, 3])
	# compute_conf_v = np.vectorize(compute_confusion_matrix, otypes=[np.ndarray], excluded=[1, 2])
	# dcf_norm_v = np.vectorize(DCF_binary, excluded=[1, 2, 3])

	# dcf_arr = (dcf_norm_v(compute_conf_v(compute_opt_v(scores, *triplet, thresholds), trueL, 2), *triplet)) / B_dummy
	# dcf_min = dcf_arr.min()

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
		dcfs = DCF_unnormalized_normalized_min_binary(llrs, trueL, (eff_pi, 1, 1))
		dcf_arr[idx] = dcfs[1]
		dcfmin_arr[idx] = dcfs[2]
	return dcf_arr, dcfmin_arr

def compute_misclassification_r(confusion_matrix):
	'''Receives a confusion matrix and computes the mis-classification ratio matrix'''
	column_sum = np.sum(confusion_matrix, axis=0)
	return confusion_matrix / column_sum # Exploit broadcasting

def DCF_unnormalized_normalized_multiclass(prior_p, misclsf_ratios, cost_matrix, norm_term):
	'''Takes formatted inputs, reshapes to 1D when necessary inside the function'''
	element_wise_mul = np.multiply(misclsf_ratios, cost_matrix) # Multiply everything element-wise
	inner_sum = np.sum(element_wise_mul, axis=0) # Sum over each row
	# ravel() to cast into 1D arrays such that np.dot returns a scalar
	dcf_u = np.dot(prior_p.ravel(), inner_sum.ravel())
	dcf_norm = dcf_u / norm_term

	return (dcf_u, dcf_norm)

def main():
	# --------------- MVG clasiffiers ---------------

	# IRIS dataset

	D, L = load('../datasets/iris.csv')
	priorP = vcol(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
	K = len(np.unique(L)) # Number of distinct classes

	idxTrain, idxTest = split_db_2to1(D)
	LTE = L[idxTest]

	gaussianL = gaussianCSF_wrapper(D, L, K, idxTrain, idxTest, priorP, log=False, show=False)
	# print(compute_confusion_matrix(gaussianL, LTE, K))

	naiveBayesGL = naiveBayesGaussianCSF(D, L, K, idxTrain, idxTest, priorP, log=False, show=False)
	# print(compute_confusion_matrix(naiveBayesGL, LTE, K))

	tiedCovGL = tiedCovarianceGaussianCSF(D, L, K, idxTrain, idxTest, priorP, log=False, show=False)
	# print(compute_confusion_matrix(tiedCovGL, LTE, K))

	tiedNaiveBayesGL = tiedNaiveBayesGaussianClassifier(D, L, K, idxTrain, idxTest, priorP, log=False, show=False)
	# print(compute_confusion_matrix(tiedNaiveBayesGL, LTE, K))

	# ----------------- Divina Commedia -------------

	divinaComediaLabels = np.load('data/commedia_labels.npy')
	print(divinaComediaLabels)
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
		(dcf_u, dcf_norm, dcf_min) = DCF_unnormalized_normalized_min_binary(llr_infpar, labels_infpar, triplet)
		print('pi_1: {}, C_fn: {}, C_fp: {}:\tDCF_u:{}\tDCF_norm: {}\tDCF_min: {}'.format(*triplet, dcf_u, dcf_norm, dcf_min))

	# ---------------------- ROC curves ----------------------

	# Outside of the for loop because this does not depend on the the application (triplet) but just on the threshold
	TPR_arr, FPR_arr, _ = ROC_TPR_vs_FPR(llr_infpar, labels_infpar)

	plt.figure('ROC')
	plt.plot(FPR_arr, TPR_arr)
	plt.grid()
	plt.xlabel('FPR')
	plt.ylabel('TPR')

	# -------------------- DET curves --------------------

	_, FPR_arr, FNR_arr = ROC_TPR_vs_FPR(llr_infpar, labels_infpar)
	plt.figure('DET')
	plt.plot(FPR_arr, FNR_arr)
	plt.grid()
	plt.xlabel('FPR')
	plt.ylabel('FNR')

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

	# -------------------- Multiclass evaluation ----------------------

	print('\nMulticlass evaluation')
	divinaCommediaLL = np.load('data/commedia_ll.npy')
	divinaCommediaLL_eps1 = np.load('data/commedia_ll_eps1.npy')
	divinaComediaLabels = np.load('data/commedia_labels.npy')
	divinaComediaLabels_eps1 = np.load('data/commedia_labels_eps1.npy')
	K = len(np.unique(divinaComediaLabels))

	cost_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
	priorP = vcol(np.array([0.3, 0.4, 0.3]))
	# priorP = vcol(np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])) # trying out as last step
	# cost_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) # trying out as last step

	posteriorP = compute_classPosteriorP(divinaCommediaLL, np.log(priorP)) # From lab 6
	posteriorP_eps1 = compute_classPosteriorP(divinaCommediaLL_eps1, np.log(priorP))

	C_vector = np.dot(cost_matrix, posteriorP)
	C_vector_eps1 = np.dot(cost_matrix, posteriorP_eps1)

	predL = np.argmin(C_vector, axis=0) # Take the minimum cost class prediction
	predL_eps1 = np.argmin(C_vector_eps1, axis=0)

	# System that only considers the prior probability and cost matrix (without looking at any sample)
	C_dummy = np.min(np.dot(cost_matrix, priorP)) # For normalization when computing DCF
	# The above system just takes cost matrix, prior probs. and computes a column vector with a "overall" cost for the classes,
	# then it just chooses the one with minimum value

	M_conf = compute_confusion_matrix(predL, divinaComediaLabels, K)
	M_conf_eps1 = compute_confusion_matrix(predL_eps1, divinaComediaLabels_eps1, K)
	R_misclsf = compute_misclassification_r(M_conf)
	R_misclsf_eps1 = compute_misclassification_r(M_conf_eps1)

	(dcf_u, dcf_norm) = DCF_unnormalized_normalized_multiclass(priorP, R_misclsf, cost_matrix, C_dummy)
	(dcf_u_eps1, dcf_norm_eps1) = DCF_unnormalized_normalized_multiclass(priorP, R_misclsf_eps1, cost_matrix, C_dummy)

	print('eps 0.001:\nM:\n', M_conf, '\nDCF_u:\n', dcf_u, '\nDCF_norm:\n', dcf_norm)
	print('eps 1:\nM:\n', M_conf_eps1, '\nDCF_u:\n', dcf_u_eps1, '\nDCF_norm:\n', dcf_norm_eps1)
	


if __name__ == '__main__':
	main()
