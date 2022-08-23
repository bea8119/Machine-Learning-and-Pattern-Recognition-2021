import matplotlib.pyplot as plt
import numpy as np
import seaborn

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