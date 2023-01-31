import utils as u
import numpy as np
import feature_utils as f
import SVM
import DCF
from LogReg import calibrate_scores

loadCalScores = False
saveCalScores = True

def main():
    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')

    # DTR, LTR = u.reduced_dataset(DTR, LTR, 4000, seed=0)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    ''' Retrieve calibration parameters from the K_fold training scores (uncalibrated) to apply them onto the test set '''

    K_fold_MVG_scores = np.load('../data_npy/scores_MVG_K_fold_PCA_None.npy')
    K_fold_SVM_scores = np.load('../data_npy/scores_SVM_K_fold_PCA_None.npy')

    np.random.seed(0)
    idx = np.random.permutation(DTR.shape[1])
    trueL_ordered = LTR[idx] # idx was computed randomly before

    calibratedMVGScores, w_MVG, b_MVG = calibrate_scores(K_fold_MVG_scores, trueL_ordered, 0.5)
    calibratedSVMScores, w_SVM, b_SVM = calibrate_scores(K_fold_SVM_scores, trueL_ordered, 0.5)

    '''
    At this point, you have a transformation pairs (w, b) to apply to the test set scores
    Get the Evaluation set scores without Calibration
    '''

    if not loadCalScores:
        testMVGScores = np.load('../data_npy/scores_MVG_Test_PCA_None.npy')
        testSVMScores = np.load('../data_npy/scores_SVM_Test_PCA_None.npy')
        calibratedMVGScores = calibrate_scores(testMVGScores, LTE, 0.5, w_MVG, b_MVG)
        calibratedSVMScores = calibrate_scores(testSVMScores, LTE, 0.5, w_SVM, b_SVM)
        if saveCalScores:
            np.save('../data_npy/scores_MVG_Test_PCA_None_calibrated.npy', calibratedMVGScores)
            np.save('../data_npy/scores_SVM_Test_PCA_None_calibrated.npy', calibratedSVMScores)
    else:
        calibratedMVGScores = np.load('../data_npy/scores_MVG_Test_PCA_None_calibrated.npy')
        calibratedSVMScores = np.load('../data_npy/scores_SVM_Test_PCA_None_calibrated.npy')
        
    # Compute the actual DCF
    print('Tied Full MVG (candidate model) (No PCA) on Test dataset after calibration with (w, b) obtained from training scores')
    for triplet in application_points:
        (_, dcf_norm, dcf_min) = DCF.DCF_unnormalized_normalized_min_binary(calibratedMVGScores, LTE, triplet)
        print(f'\tpi_eff = {triplet[0]}\tminDCF: {round(dcf_min, 3)}    actDCF: {round(dcf_norm, 3)}')

    print()

    print('Quad SVM (candidate model) (No PCA) on Test dataset after calibration with (w, b) obtained from training scores')
    for triplet in application_points:
        (_, dcf_norm, dcf_min) = DCF.DCF_unnormalized_normalized_min_binary(calibratedSVMScores, LTE, triplet)
        print(f'\tpi_eff = {triplet[0]}\tminDCF: {round(dcf_min, 3)}    actDCF: {round(dcf_norm, 3)}')


if __name__ == '__main__':
    main()