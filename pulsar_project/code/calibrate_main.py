import utils as u
import numpy as np
import feature_utils as f
import SVM
from LogReg import calibrate_scores

def main():
    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')

    DTR, LTR = u.reduced_dataset(DTR, LTR, 4000, seed=0)

    D_merged, L_merged, idxTR_merged, idxTE_merged = u.split_db_after_merge(DTR, DTE, LTR, LTE) # Merged split

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    priorT_b = 0.5

    w, b = SVM.SVM_wrapper(D_merged, L_merged, 1, 0.1, priorT_b, idxTR_merged, idxTE_merged, application_points[0], 1, 2, 1e-1, True,
        False, True, True, calibrate=True, saveCalibration=True)

    # At this point, you have a transformation (w, b) to apply to the Evaluation set scores
    # Get the Evaluation set scores without Calibration

    testScores = np.load('../data_npy/scores_SVM_Test_PCA_None.npy')
    calibratedScores = calibrate_scores(testScores, LTE, 0.5, w, b)

    # Compute the actual DCF
    # ...

if __name__ == '__main__':
    main()