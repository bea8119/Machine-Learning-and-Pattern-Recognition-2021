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

def Z_normalization(D):
    '''Returns Z_D (Z-transformed dataset) and the corresponding Mean and Standard Deviation'''
    mean_vector = vcol(np.mean(D, axis=1))
    std_vector = vcol(np.std(D, axis=1))

    Z_D = (D - mean_vector) / std_vector
    
    return Z_D
