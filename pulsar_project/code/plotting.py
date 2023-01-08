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
        plt.grid(visible=True)
    # plt.show()

def plotHeatmap(D, L):
    plt.figure('Whole Dataset')
    seaborn.heatmap(np.abs(np.corrcoef(D)), linewidth=0.5, cmap="Greys", square=True, cbar=False)
    plt.xlabel('Whole Dataset')
    plt.figure('False class samples')
    seaborn.heatmap(np.abs(np.corrcoef(D[:, L == 0])), linewidth=0.5, cmap="Reds", square=True, cbar=False)
    plt.xlabel('False Class Samples')
    plt.figure('True class samples')
    seaborn.heatmap(np.abs(np.corrcoef(D[:, L == 1])), linewidth=0.5, cmap="Blues", square=True, cbar=False)
    plt.xlabel('True Class Samples')
    # plt.show()

def plotDCFmin_vs_lambda(l_arr, min_DCF_single_arr, min_DCF_kfold_arr, m_PCA, n, K, colors, eff_priors, quad=False, save_fig=False):
    '''Receives 3 arrays to plot (curves) for each eff_prior'''
    if min_DCF_single_arr:
        fig_name_single = 'Single-fold ({}-to-1) {} Log Reg -> lambda tuning {}'.format(
            n, 'Quadratic' if quad else 'Linear', '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_single = plt.figure(fig_name_single)
    if min_DCF_kfold_arr:
        fig_name_kfold = '{}-fold {} Log Reg -> lambda tuning {}'.format(
            K, 'Quadratic' if quad else 'Linear', '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_kfold = plt.figure(fig_name_kfold)

    for i in range(len(eff_priors)):
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.plot(l_arr, min_DCF_single_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(l_arr), max(l_arr)])
            plt.xscale('log')
            plt.xlabel(r'$\lambda$')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.plot(l_arr, min_DCF_kfold_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(l_arr), max(l_arr)])
            plt.xscale('log')
            plt.xlabel(r'$\lambda$')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
    
    if save_fig:
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.savefig('../plots/' + fig_name_single.replace('>', '-') + '.png')
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.savefig('../plots/' + fig_name_kfold.replace('>', '-') + '.png')

def plotDCFmin_vs_C_linearSVM(C_arr, min_DCF_single_arr, min_DCF_kfold_arr, pi_b, m_PCA, n, K, colors, eff_priors, save_fig=False):
    '''Tuning of C parameter alone, on every application point'''
    if min_DCF_single_arr:
        fig_name_single = 'Single-fold ({}-to-1) Linear SVM -> C tuning (pi_T = {}) {}'.format(
            n, 'unbalanced' if pi_b is None else pi_b, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_single = plt.figure(fig_name_single)
    if min_DCF_kfold_arr:
        fig_name_kfold = '{}-fold Linear SVM -> C tuning (pi_T = {}) {}'.format(
            K, 'unbalanced' if pi_b is None else pi_b, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_kfold = plt.figure(fig_name_kfold)

    for i in range(len(eff_priors)):
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.plot(C_arr, min_DCF_single_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.plot(C_arr, min_DCF_kfold_arr[i], color=colors[i], label='min DCF {} = {}'.format(r'$\tilde{\pi}$', eff_priors[i][0]))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)

    if save_fig:
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.savefig('../plots/' + fig_name_single.replace('>', '-') + '.png')
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.savefig('../plots/' + fig_name_kfold.replace('>', '-') + '.png')

def plotDCFmin_vs_C_quadSVM(C_arr, min_DCF_single_arr, min_DCF_kfold_arr, m_PCA, n, K, colors, app_point, c_list, save_fig=False):
    '''Tuning of C jointly with c (in linear scale), take three different values of c on the same application point'''
    if min_DCF_single_arr:
        fig_name_single = 'Single-fold ({}-to-1) Quadratic Kernel SVM -> C - c tuning {}'.format(
            n, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_single = plt.figure(fig_name_single)
    if min_DCF_kfold_arr:
        fig_name_kfold = '{}-fold Quadratic Kernel SVM -> C - c tuning {}'.format(
            K, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_kfold = plt.figure(fig_name_kfold)
    
    for i in range(len(c_list)):
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.plot(C_arr, min_DCF_single_arr[i], color=colors[i], label='min DCF {} = {}, c = {}'.format(
                r'$\tilde{\pi}$', app_point[0], c_list[i]))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.plot(C_arr, min_DCF_kfold_arr[i], color=colors[i], label='min DCF {} = {}, c = {}'.format(
                r'$\tilde{\pi}$', app_point[0], c_list[i]))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)

    if save_fig:
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.savefig('../plots/' + fig_name_single.replace('>', '-') + '.png')
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.savefig('../plots/' + fig_name_kfold.replace('>', '-') + '.png')

def plotDCFmin_vs_C_RBFSVM(C_arr, min_DCF_single_arr, min_DCF_kfold_arr, m_PCA, n, K, colors, app_point, gamma_list, save_fig=False):
    '''Tuning of C jointly with gamma (in log scale), take different values of gamma on the same application point'''
    if min_DCF_single_arr:
        fig_name_single = 'Single-fold ({}-to-1) RBF Kernel SVM -> C - gamma tuning {}'.format(
            n, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_single = plt.figure(fig_name_single)
    if min_DCF_kfold_arr:
        fig_name_kfold = '{}-fold RBF Kernel SVM -> C - gamma tuning {}'.format(
            K, '(no PCA)' if m_PCA is None else f'(PCA m = {m_PCA})')
        fig_kfold = plt.figure(fig_name_kfold)
    
    for i in range(len(gamma_list)):
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.plot(C_arr, min_DCF_single_arr[i], color=colors[i], label='min DCF {} = {}, log {} = {}'.format(
                r'$\tilde{\pi}$', app_point[0], r'$\gamma$', int(np.log10(gamma_list[i]))))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.plot(C_arr, min_DCF_kfold_arr[i], color=colors[i], label='min DCF {} = {}, log {} = {}'.format(
                r'$\tilde{\pi}$', app_point[0], r'$\gamma$', int(np.log10(gamma_list[i]))))
            plt.xlim([min(C_arr), max(C_arr)])
            plt.xscale('log')
            plt.xlabel('C')
            plt.ylabel('DCF')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.grid(visible=True)

    if save_fig:
        if min_DCF_single_arr:
            plt.figure(fig_single)
            plt.savefig('../plots/' + fig_name_single.replace('>', '-') + '.png')
        if min_DCF_kfold_arr:
            plt.figure(fig_kfold)
            plt.savefig('../plots/' + fig_name_kfold.replace('>', '-') + '.png')

def create_GMM_figure(tied, diag):
    '''Receives tied and diag flags and n=4, returns a list of figure objects with the appropriate names'''
    GMM_type = ''
    if tied:
        GMM_type += 'Tied '
    if diag:
        GMM_type += 'Diag '
    if not diag and not tied:
        GMM_type += 'Full '
    GMM_type += 'Covariance '

    return plt.figure(GMM_type)

def plotGMM(n_splits, dcf_min_list, eff_prior, tied_diag_pairs, colors, PCA_list):

    one = 1

    for i, (tied, diag) in enumerate(tied_diag_pairs):
        GMM_type = ''
        if tied:
            GMM_type += 'Tied '
        if diag:
            GMM_type += 'Diag '
        if not diag and not tied:
            GMM_type += 'Full '
        GMM_type += 'Covariance '

        plt.figure('{}GMM classifier'.format(GMM_type))
        for j, m in enumerate(PCA_list):
            plt.bar(np.arange(1, n_splits + 1) + 0.1 * one, dcf_min_list[j][i], label='min DCF {} = {} {}'.format(
                r'$\tilde{\pi}$', eff_prior, '(no PCA)' if m is None else '(PCA m = {})'.format(m)),
                color=colors[j], width=0.2)
            one *= -1
        plt.xlabel('Number of components')
        plt.xticks(np.arange(1, n_splits + 1), 2**np.arange(1, n_splits + 1))
        plt.ylabel('DCF')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(visible=True)
    

def main():

    DTR, LTR = u.load('../data/Train.txt')
    DTE, LTE = u.load('../data/Test.txt')
    
    # Pre-processing (Z-normalization)
    DTR, mean, std = f.Z_normalization(DTR)
    # DTE = f.Z_normalization(DTE, mean, std)

    # Plot distribution of attribute values (after Z-Normalizing) for each class
    plotHistogram(DTR, LTR, CLASS_NAMES, ATTRIBUTE_NAMES)

    # Plot heatmap of covariance
    plotHeatmap(DTR, LTR)
    plt.show()
    plt.savefig()

if __name__ == '__main__':
    main()
