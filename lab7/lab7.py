import numpy as np
import sys
import scipy.optimize
import sklearn.datasets

# sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from utility.vrow_vcol import vcol, vrow
from lab2.load_plot import load
from lab5.lab5 import split_db_2to1, split_dataset

# 2 methods: 
# 1) Numerical approx. of the gradient of f (MUCH more expensive)
# 2) Explicitly passed gradient (solved computing the gradient of f manually)
#   - Better

def f_1(y_z):
    return (y_z[0] + 3)**2 + np.sin(y_z[0]) + (y_z[1] + 1)**2

def f_2(y_z):
    # Returns a tuple containing f and gradient(f) (which is a (2,) size ndarray)
    return (
        (y_z[0] + 3)**2 + np.sin(y_z[0]) + (y_z[1] + 1)**2,
        np.array([2 * (y_z[0] + 3) + np.cos(y_z[0]), 2 * (y_z[1] + 1)])
        )

def load_iris_binary():
    D, L = load('../datasets/iris.csv')
    # D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L != 0] # We remove setosa from L
    L[L == 2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def computeAccuracy_logreg_binary(scoreArray, TrueL):
    PredictedL = np.array([(1 if score > 0 else 0) for score in scoreArray.ravel()])
    NCorrect = (PredictedL.ravel() == TrueL.ravel()).sum() # Will count as 1 the "True"
    NTotal = TrueL.size
    return float(NCorrect) / float(NTotal)

class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
    def logreg_obj_binary(self, v):
    # Compute and return the objective function value. You can retrieve all required information from self.DTR, self.LTR, self.l
        # v is an ndarray of shape (D+1,) packs model parameters [w (of size D), b]
        w, b = vcol(v[0:-1]), v[-1]
        s = np.dot(w.T, self.DTR) + b
        z = 2.0 * self.LTR - 1

        mean_term = np.logaddexp(0, -z * s).mean()
        return 0.5 * self.l * np.linalg.norm(w)**2 + mean_term

    def logreg_obj_multiclass(self, v):
        w, b = v[0:-1], v[-1]


def main():
    # ----------- Numerical optimization ---------------
    # print('Method 1: Approximated gradient (should be more expensive)')
    # (x, f, d) = scipy.optimize.fmin_l_bfgs_b(f_1, np.array([0, 0]), approx_grad=True) # 21 funcalls
    # print(x)
    # print(f)
    # print(d)

    # print('Method 2: Explicitly passed gradient (should be less expensive)')
    # (x, f, d) = scipy.optimize.fmin_l_bfgs_b(f_2, np.array([0, 0])) # 7 funcalls
    # # print(x)
    # # print(f)
    # # print(d)
    # --------------------------------------------------

    # ------------- Binary logistic regression --------------- 
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, *split_db_2to1(D))

    # l stands for lambda
    l = 1e-0

    logRegObj = logRegClass(DTR, LTR, l)

    x0 = np.zeros(DTR.shape[0] + 1)
    (v, J, d) = scipy.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj_binary, x0, approx_grad=True)
    print(v)
    print(J)
    print(d)
    # v stores the wrapped up values of w and b (w is of size 4 for IRIS dataset (dim of feature space))
    w = vcol(v[0:-1])
    b = v[-1]
    scoreArray = np.dot(w.T, DTE) + b
    accuracy = computeAccuracy_logreg_binary(scoreArray, LTE)
    err_rate = 1 - accuracy
    print('Error rate with lambda', l, ':', round(err_rate * 100, 1), '%')

    # ----------------------------
    
    # ---------------- Multiclass logistic regression ---------------
    D, L = load('../datasets/iris.csv')
    (DTR, LTR), (DTE, LTE) = split_dataset(D, L, *split_db_2to1(D))

    l = 1e-0
    logRegObj = logRegClass(DTR, LTR, l)

if __name__ == '__main__':
    main()
    
