# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:38:37 2022

@author: shado

Linear SVM
"""
#Imports
import sys
import numpy 
import sklearn.datasets
import scipy
import scipy.special 

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[ L != 0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

#-------------BASIC FUNCTIONS------------------------
def mrow(x):
    return x.reshape((1,x.size))

def mcol(v):
    return v.reshape((v.size,1))

def compute_empirical_mean(X):
    return mcol(X.mean(1))

def compute_empirical_cov(X):
    mu=compute_empirical_mean(X)
    cov=numpy.dot((X-mu),(X-mu).T)/X.shape[1]
    return cov

#---- SPLIT 2-TO-1-----------------------------------------
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

def train_SVM_linear(DTR,LTR,C,K=1):
    DTREXT=numpy.vstack([DTR,numpy.ones((1,DTR.shape[1]))])
    
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1]=1
    Z[LTR==0]=-1
    H=numpy.dot(DTREXT.T,DTREXT)
    #Dist=mcol((DTR**2.sum(0))+mrow((DTR**2).sum(0))-2*numpy.dot(DTR.T,DTR))
    #H=numpy.exp(-Dist)
    H=mcol(Z)*mrow(Z)*H
    
    def JDual(alpha):
        Ha=numpy.dot(H,mcol(alpha))
        aHa=numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        return -0.5 * aHa.ravel()+a1, -Ha.ravel() + numpy.ones(alpha.size)
    
    def LDual(alpha):
        loss,grad=JDual(alpha)
        return -loss,-grad
    
    def JPrimal(w):
        S=numpy.dot(mrow(w),DTREXT)
        print(S)
        loss=numpy.maximum(numpy.zeros(S.shape),1-Z*S).sum()
        print(loss)
        return 0.5 * numpy.linalg.norm(w)**2+C*loss
    
    alphaStar,_x,_y=scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds=[(0,C)]*DTR.shape[1],
        factr=1.0
        )
    # print(alphaStar)
    
    wStar=numpy.dot(DTREXT,mcol(alphaStar)*mcol(Z))
    # print(wStar)

    print(JPrimal(wStar) - JDual(alphaStar)[0])
    
D,L=load_iris_binary()
(DTR,LTR),(DTE,LTE)=split_db_2to1(D,L)

train_SVM_linear(DTR,LTR,1.0,1.0)
sys.exit(0)