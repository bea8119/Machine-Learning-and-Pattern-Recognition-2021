import numpy as np
import sys
import scipy.special

# sys.path.append('/home/oormacheah/Desktop/Uni shit/MLPR') # for linux
sys.path.append('C:/Users/andre/Desktop/Cositas/poli_repo/MLPR_21-22') # for windows
from utility.vrow_vcol import vcol, vrow
from load import load_data, split_data

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

def buildLogFrequencies_withEPS(hlTercets, eps=0.001):
    ''' 
    Returns Python dictionary whose keys are the classes. For each class, h_clsLogProb[cls] is a dictionary whose 
    keys are words and values are the corresponding log-frequencies (model parameters for class cls)
    '''
    commonWordSet = set([]) # Set of all words (independently of the Cantica)

    # Iterate through classes (3 iterations/classes)
    # Initialize with pseudo-count (eps) each word
    for cls in hlTercets:
        cls_wordSet = buildWordSet(hlTercets[cls])
        commonWordSet = commonWordSet.union(cls_wordSet)
    
    # Initialize (with eps) and compute the counts of words for each class
    h_clsLogProb = {}
    for cls in hlTercets:
        h_clsLogProb[cls] = {word: eps for word in commonWordSet}
        for tercet in hlTercets[cls]:
            for word in tercet.split():
                h_clsLogProb[cls][word] += 1

    # h_clsLogProb[cls] contains word keys only and its "count" (real count + pseudo-count) as values

    # Now, compute the log frequencies
    for cls in hlTercets:
        nWordsCls = sum(h_clsLogProb[cls].values()) # Sum of occurrencies of ALL words of the Cantica class
        for w in h_clsLogProb[cls]:
            h_clsLogProb[cls][w] = np.log(h_clsLogProb[cls][w]) - np.log(nWordsCls) # Compute log N_{cls,w} / N_cls
    
    return h_clsLogProb

def compute_logLikelihoods(h_clsLogProb, tercet):
    # Initialize logLikelihoodCls (class-conditional) with 0
    logLikelihoodCls = {cls: 0 for cls in h_clsLogProb} # 3 classes at first, so a dictionary with 3 keys

    for cls in h_clsLogProb: # Loop over classes
        for word in tercet.split(): # Loop over words (approach 1 (X), doesn't consider every word of the dictionary)
            if word in h_clsLogProb[cls]: # Ignore words that are not in the common dictionary
                logLikelihoodCls[cls] += h_clsLogProb[cls][word]
    return logLikelihoodCls # Python dictionary

def compute_logLikelihoodMatrix(h_clsLogProb, lTercets, hCls2Idx):

    S = np.zeros((len(h_clsLogProb), len(lTercets)))

    for tercetIndex, tercet in enumerate(lTercets):
        hScores = compute_logLikelihoods(h_clsLogProb, tercet)
        for cls in h_clsLogProb:
            clsIdxInt = hCls2Idx[cls]
            S[clsIdxInt, tercetIndex] = hScores[cls]

    return S

def compute_classPosteriorP(S, logPrior):
    logSJoint = S + logPrior

    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0)) # compute for each column

    logSPost = logSJoint - logSMarginal

    SPost_afterLog = np.exp(logSPost)

    return SPost_afterLog

def compute_accuracy(Posteriors, TrueL):
    PredictedL = np.argmax(Posteriors, axis=0) # INDEX of max through axis 0 (per sample)
    NCorrect = (PredictedL.ravel() == TrueL.ravel()).sum() # Will count as 1 the "True"
    NTotal = TrueL.size
    return float(NCorrect)/float(NTotal)

if __name__ == '__main__':
    lInf, lPur, lPar = load_data()

    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    hlTercetsTrain = {
        'inferno': lInf_train,
        'purgatorio': lPur_train,
        'paradiso': lPar_train
    }

    modelDict = buildLogFrequencies_withEPS(hlTercetsTrain)

    hCls2Idx = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    lTercetsEval = lInf_evaluation + lPur_evaluation + lPar_evaluation

    scoreMatrix = compute_logLikelihoodMatrix(modelDict, lTercetsEval, hCls2Idx)

    priorP_log = vcol(np.log(np.array([1./3., 1./3., 1./3.])))

    posteriorP = compute_classPosteriorP(scoreMatrix, priorP_log)

    labelsInf = np.zeros(len(lInf_evaluation))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = np.zeros(len(lPar_evaluation))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsPur = np.zeros(len(lPur_evaluation))
    labelsPur[:] = hCls2Idx['purgatorio']

    labelsEval = np.hstack([labelsInf, labelsPur, labelsPar])

    acc_inferno = compute_accuracy(posteriorP[:, labelsEval == hCls2Idx['inferno']], labelsEval[labelsEval == hCls2Idx['inferno']])
    print('Multiclass - Inferno - Accuracy:', str(round(acc_inferno * 100)), '%')

    acc_purgatorio = compute_accuracy(posteriorP[:, labelsEval == hCls2Idx['purgatorio']], labelsEval[labelsEval == hCls2Idx['purgatorio']])
    print('Multiclass - Purgatorio - Accuracy:', str(round(acc_purgatorio * 100)), '%')

    acc_paradiso = compute_accuracy(posteriorP[:, labelsEval == hCls2Idx['paradiso']], labelsEval[labelsEval == hCls2Idx['paradiso']])
    print('Multiclass - Paradiso - Accuracy:', str(round(acc_paradiso * 100)), '%')

    # ---------- End of solution --------- #

    # Binary pair case (inferno - paradiso) RE-TRAINING the model

    hCls2Idx = {'inferno': 0, 'paradiso': 1}

    hlTercetsTrain = {
        'inferno': lInf_train,
        'paradiso': lPar_train
        }

    lTercetsEval = lInf_evaluation + lPar_evaluation

    S1_model = buildLogFrequencies_withEPS(hlTercetsTrain, eps = 0.001)

    S1_predictions = compute_classPosteriorP(
        compute_logLikelihoodMatrix(
            S1_model,
            lTercetsEval,
            hCls2Idx,
            ),
        vcol(np.log(np.array([1./2., 1./2.])))
        )

    labelsInf = np.zeros(len(lInf_evaluation))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = np.zeros(len(lPar_evaluation))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsEval = np.hstack([labelsInf, labelsPar])

    print('Binary [inferno vs paradiso] - Accuracy:', str(round(compute_accuracy(S1_predictions, labelsEval) * 100)), '%')