import utils as u
import feature_utils as f
import MVG
import LogReg as LR
import numpy as np
import scipy.optimize

def main():
    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')
    
    # Pre-processing (Z-normalization)
    DTR = f.Z_normalization(DTR)
    DTE = f.Z_normalization(DTE)

    application_points = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]

    # ----------------- Logistic Regression -------------------

    l = 1e-6 # Lambda hyperparameter
    logRegObj = LR.logRegClass(DTR, LTR, l)

    x0 = np.zeros(DTR.shape[0] + 1)

    (v, J, d) = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj_binary, x0, approx_grad=True)

if __name__ == '__main__':
    main()