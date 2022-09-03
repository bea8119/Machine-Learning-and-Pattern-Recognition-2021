from utils import vrow
from LogReg import logRegClass
import numpy as np

def calibrate_scores(scores_D, L, eff_prior):
    '''This takes the scores (1D) of the classifiers (whatever the model yields as scores) and returns
    the calibrated scores after applying a LogReg with lambda = 0 (no regularization) and a given effective
    prior to use as the "prior" of the LogReg model. This LogReg will return a new set of scores (applying the 
    optimal w and b to the same scores used for training the data) that are the transformed (hopefully optimal)
    scores.'''
    scores_D = vrow(scores_D)
    logRegObj = logRegClass(scores_D, L, 0, eff_prior)
    logreg_scores, w, b = logRegObj.logreg_scores(scores_D, calibrate=True)
    calibrated_scores = logreg_scores - np.log(eff_prior / (1 - eff_prior))

    return calibrated_scores, w, b