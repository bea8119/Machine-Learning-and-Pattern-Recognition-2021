from utils import load
from feature_utils import Z_normalization, PCA_givenM
from plotting import plotHeatmap, plotHistogram
import matplotlib.pyplot as plt

CLASS_NAMES = ['RFI / Noise', 'Pulsar']
ATTRIBUTE_NAMES = ['Mean of the integrated profile',
                 'Standard deviation of the integrated profile',
                 'Excess kurtosis of the integrated profile',
                 'Skewness of the integrated profile',
                 'Mean of the DM-SNR curve',
                 'Standard deviation of the DM-SNR curve',
                 'Excess kurtosis of the DM-SNR curve',
                 'Skewness of the DM-SNR curve']

def main():

    (DTR, LTR) = load('../data/Train.txt')
    (DTE, LTE) = load('../data/Test.txt')
    Z_DTR = Z_normalization(DTR)

    # Plot distribution of attribute values (after Z-Normalizing) for each class
    plotHistogram(Z_DTR, LTR, CLASS_NAMES, ATTRIBUTE_NAMES)

    # Plot heatmap of covariance
    plotHeatmap(DTR, LTR)
    plt.show()

    # Apply PCA
    M = 6
    PCA_ProjM = PCA_givenM(Z_DTR, M)
    


if __name__  == '__main__':
    main()
    