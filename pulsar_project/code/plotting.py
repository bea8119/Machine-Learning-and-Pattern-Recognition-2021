import utils as u
import feature_utils as f
import matplotlib.pyplot as plt
import numpy as np
import seaborn

CLASS_NAMES = ['RFI / Noise', 'Pulsar']
ATTRIBUTE_NAMES = ['Mean of the integrated profile',
                 'Standard deviation of the integrated profile',
                 'Excess kurtosis of the integrated profile',
                 'Skewness of the integrated profile',
                 'Mean of the DM-SNR curve',
                 'Standard deviation of the DM-SNR curve',
                 'Excess kurtosis of the DM-SNR curve',
                 'Skewness of the DM-SNR curve']

def plotHistogram(D, L, class_names, attribute_names):
    # dataset for each class
    d0 = D[:, L == 0]
    d1 = D[:, L == 1]
    
    n_features = D.shape[0]
    
    alpha = 0.6 # Opacity coefficient
    bins = 70 # N_bins

    # now, divide per value type in a loop and plot a figure containing all the classes

    for i in range(n_features):
        plt.figure(attribute_names[i])
        plt.hist(d0[i, :], bins=bins, density=True, alpha=alpha, label=class_names[0], color='r', ec='black')
        plt.hist(d1[i, :], bins=bins, density=True, alpha=alpha, label=class_names[1], color='b', ec='black')
        plt.xlabel(attribute_names[i])
        plt.legend()
        plt.tight_layout()
        plt.grid()
    # plt.show()

def plotHeatmap(D, L):
    plt.figure('Whole Dataset')
    seaborn.heatmap(np.abs(np.corrcoef(D)), linewidth=0.5, cmap="Greys", square=True, cbar=False)
    plt.figure('False class samples')
    seaborn.heatmap(np.abs(np.corrcoef(D[:, L == 0])), linewidth=0.5, cmap="Reds", square=True, cbar=False)
    plt.figure('True class samples')
    seaborn.heatmap(np.abs(np.corrcoef(D[:, L == 1])), linewidth=0.5, cmap="Blues", square=True, cbar=False)
    # plt.show()

def plotDCFmin_vs_lambda(l_arr, min_DCF_single_arr, min_DCF_kfold_arr, m_PCA, n, K, colors, eff_priors):
    '''Receives 3 arrays to plot (curves) for each eff_prior'''

    fig_single = plt.figure('Single-fold ({}-to-1) lambda tuning {}'.format(n, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})'))
    fig_kfold = plt.figure('{}-fold lambda tuning {}'.format(K, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})'))

    for i in range(len(eff_priors)):
        plt.figure(fig_single)
        plt.plot(l_arr, min_DCF_single_arr[i], color=colors[i], label=f'eff_prior = {eff_priors[i][0]}')
        plt.xlim([min(l_arr), max(l_arr)])
        plt.xscale('log')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('min DCF')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid()

        if min_DCF_kfold_arr is not None:
            plt.figure(fig_kfold)
            plt.plot(l_arr, min_DCF_kfold_arr[i], color=colors[i], label=f'eff_prior = {eff_priors[i][0]}')
            plt.xlim([min(l_arr), max(l_arr)])
            plt.xscale('log')
            plt.xlabel(r'$\lambda$')
            plt.ylabel('min DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid()

def main():

    DTR, LTR = u.load('../data/Train.txt')
    # DTE, LTE = u.load('../data/Test.txt')
    
    # Pre-processing (Z-normalization)
    DTR, mean, std = f.Z_normalization(DTR)
    DTE = f.Z_normalization(DTE, mean, std)

    # Plot distribution of attribute values (after Z-Normalizing) for each class
    plotHistogram(DTR, LTR, CLASS_NAMES, ATTRIBUTE_NAMES)

    # Plot heatmap of covariance
    plotHeatmap(DTR, LTR)
    plt.show()

if __name__ == '__main__':
    main()
