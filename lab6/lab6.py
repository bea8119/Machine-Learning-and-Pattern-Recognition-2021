import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.special

# sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from lab3.lab3 import SW_compute, datasetCovarianceM, SB_compute
from utility.vrow_vcol import vcol, vrow
from lab4.lab4 import logpdf_GAU_ND
from lab6.load import load_data, split_data

def buildWordSet(lTercets):
    # Receives a List of Tercets (remember that each Row of text is a Tercet) and returns a Python SET that contains
    # all the words for the passed list of tercets without repetition, to be converted later on in a Python dict.
    # with the set entries as keys and values as the occurrence expressions

    # Avoid duplication exploting Python sets
    wordSet = set([])
    for row in lTercets:
        words = row.split()
        for word in words:
            wordSet.add(word)
    return wordSet



if __name__ == '__main__':
    lInf, lPur, lPar = load_data()

    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)


    

