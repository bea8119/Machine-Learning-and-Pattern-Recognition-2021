import numpy as np

def vcol(oneDarray):
    return oneDarray.reshape((oneDarray.size, 1))

def vrow(oneDarray):
    return oneDarray.reshape((1, oneDarray.size))

def load(filename):
    f = open(filename, 'r')
    feature_size = len(f.readline().rstrip().split(',')) - 1
    f.seek(0)
    datasetList = np.array([]).reshape((feature_size, 0))
    labelVector = []

    for line in f:
        line = line.rstrip()
        valueList = line.split(',')
        entryList = np.array([float(i) for i in valueList[:-1]]).reshape((feature_size, 1))
        datasetList = np.hstack([datasetList, entryList])
        labelVector.append(int(valueList[-1]))
    f.close()
    labelVector = np.array(labelVector) # 1D array
    return datasetList, labelVector

def split_db_n_to_1(D, n, seed=0):
    '''Returns idxTrain and idxTest according to n-to-1 splitting (n is given by the user)'''
    nTrain = int(D.shape[1] * float(n) / float(n + 1))
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) # take a random order of indexes from 0 to N
    idxTrain = idx[0:nTrain] 
    idxTest = idx[nTrain:]
    return idxTrain, idxTest

def split_dataset(D, L, idxTrain, idxTest):
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def split_db_after_merge(DTR, DTE, LTR, LTE):
    '''Returns merged dataset (as if Train and Test data were a single dataset) and the corresponding indexes'''
    D_merged = np.hstack((DTR, DTE))
    L_merged = np.concatenate((LTR, LTE))
    idxTrain = np.arange(0, DTR.shape[1])
    idxTest = np.arange(DTR.shape[1], DTR.shape[1] + DTE.shape[1])
    return D_merged, L_merged, idxTrain, idxTest 

def reduced_dataset(D, L, N, seed=0):
    '''For test purposes. Receives a dataset, its labels and an integer number N. 
    Returns a reduced dataset and labels (of N samples) randomly sampled from the given one'''
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) # take a random order of indexes from 0 to N
    idx_trunc = idx[:N]
    return D[:, idx_trunc], L[idx_trunc]