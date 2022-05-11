import matplotlib.pyplot as plt
import numpy as np

def load(filename):
    f = open(filename, 'r')
    datasetList = np.array([]).reshape((4, 0))
    labelVector = []
    labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    for line in f:
        line = line.rstrip()
        valueList = line.split(',')
        entryList = np.array([float(i) for i in valueList[:4]]).reshape((4, 1))
        datasetList = np.hstack([datasetList, entryList])
        labelVector.append(labels[valueList[4]])
    f.close()
    labelVector = np.array(labelVector)
    return datasetList, labelVector

def plotHistogram(dataset, labels):
    # masks
    m0 = labels == 0
    m1 = labels == 1
    m2 = labels == 2

    # dataset for each class
    d0 = dataset[:, m0]
    d1 = dataset[:, m1]
    d2 = dataset[:, m2]
    
    # dictionary for mapping measure plot xlabel to number (length -> 0, width -> 1, etc...)
    measureIndex = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    # now, divide per value type in a loop and plot a figure containing all the classes

    for field in range(4):
        plt.figure(measureIndex[field])
        plt.xlabel(measureIndex[field])
        plt.hist(d0[field, :], bins=10, density=True, alpha=0.4, label='Setosa')
        plt.hist(d1[field, :], bins=10, density=True, alpha=0.4, label='Versicolor')
        plt.hist(d2[field, :], bins=10, density=True, alpha=0.4, label='Virginica')
        plt.grid()
        plt.legend()
    plt.show()

def plotScatter(D, L):
    return      

dataset, labels = load('iris.csv')
plotHistogram(dataset, labels)
