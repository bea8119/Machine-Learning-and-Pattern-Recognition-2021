# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:06:35 2022

@author: shado
"""

import sys
import scipy.optimize 
import numpy 
import sklearn
import sklearn.datasets

"""
Numerical optimization
we have to use scipy.optimize.fmin_l_bfgs_b(func,x0)
where func is the function we want to minimize and x0 is the starting value for the algorithm

f(y,z)=(y+3)^2+sin(y)+(z+1)^2
"""

#list1=[200,201]
#parameters=numpy.array(list1)
#print(parameters.shape) #(2,), ok

def f(anArray):
    return (anArray[0]+3)**2+numpy.sin(anArray[0])+(anArray[1]+1)**2

_x,_f,_d=scipy.optimize.fmin_l_bfgs_b(f,numpy.zeros(2),approx_grad=True)
print(_x) #2d array that is the solution
print(_f) #value of the objective function at the minimum
print(_d) #additional information

def f(x):
    y,z=x[0],x[1]
    obj=(y+3)**2+numpy.sin(y)+(z+1)**2
    grad=numpy.array([2*(y+3)+numpy.cos(y),2*(z+1)]) #i provide an  explicit gradient
    return obj,grad

_x,_f,_d=scipy.optimize.fmin_l_bfgs_b(f,numpy.zeros(2),approx_grad=False)
print(_x) #2d array that is the solution
print(_f) #value of the objective function at the minimum
print(_d) #additional information

#-------------- BINARY LOGISTIC REGRESSION ------------------
def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[ L != 0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def logreg_obj_wrap(DTR, LTR, l):
    Z=LTR * 2.0 - 1.0
    M = DTR.shape[0]
    def logreg_obj(v): #v packs w b 
        w=mcol(v[0:M])
        b=v[-1]
        #DTR is a matrix of array [x1...xn]
        S=numpy.dot(w.T,DTR) + b #s_c 
        #i need to compute the log(1+e^sc)
        cxe=numpy.logaddexp(0,-S*Z).mean() #Z=[Z0...Zx-1] S=[S0...Sx-1] apply to all element
        #I can do also with a fo
        """
        cxe=0
        for i in range (DTR.shape[1]):
            x=DTR[:,i:i+1]
            s=numpy.dot(w.T,x)+b
            cxe+=numpy.logaddex
        """
        return cxe+0.5*l*numpy.linalg.norm(w)**2
    return logreg_obj

if __name__=='__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    for lamb in [1e-6,1e-3,0.1,1.0]:
        logreg_obj=logreg_obj_wrap(DTR,LTR,lamb)
        _v,_J,_d=scipy.optimize.fmin_l_bfgs_b(logreg_obj_wrap(DTR,LTR,lamb),numpy.zeros(DTR.shape[0]+1), approx_grad=True)
        _w=_v[0:DTR.shape[0]]
        _b=_v[-1]
        STE=numpy.dot(_w.T,DTE)+_b
        LP=STE>0
        print(lamb,_J)

