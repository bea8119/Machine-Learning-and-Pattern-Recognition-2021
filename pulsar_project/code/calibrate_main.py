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

    D_merged, L_merged, idxTR_merged, idxTE_merged = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    ''' Retrieve calibration parameters from the K_fold training scores to apply them onto the test set '''
    K_fold_SVM_scores = np.load('../data_npy/scores_SVM_K_fold_PCA_None_calibrated.npy')

    np.random.seed(0)
    idx = np.random.permutation(DTR.shape[1])
    trueL_ordered = LTR[idx] # idx was computed randomly before

    calibratedScores, w, b = calibrate_scores(K_fold_SVM_scores, trueL_ordered, 0.5)

    '''
    At this point, you have a transformation (w, b) to apply to the test set scores
    Get the Evaluation set scores without Calibration
    '''

    if not loadCalScores:
        testScores = np.load('../data_npy/scores_SVM_Test_PCA_None.npy')
        calibratedScores = calibrate_scores(testScores, LTE, 0.5, w, b)
        if saveCalScores:
            np.save('../data_npy/scores_SVM_Test_PCA_None_calibrated.npy', calibratedScores)
    else:
        calibratedScores = np.load('../data_npy/scores_SVM_Test_PCA_None_calibrated.npy')
        
    # Compute the actual DCF
    print('Quad SVM (candidate model) (No PCA) on Test dataset')
    for triplet in application_points:
        (_, dcf_norm, dcf_min) = DCF.DCF_unnormalized_normalized_min_binary(calibratedScores, LTE, triplet)
        print(f'pi_eff = {triplet[0]}\tminDCF: {dcf_min}    actDCF: {dcf_norm}')


if __name__ == '__main__':
    main()