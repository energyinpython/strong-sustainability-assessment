import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyrepo_mcda.additions import rank_preferences
from ssp_vikor import SSP_VIKOR
from ssp_spotis import SPOTIS_SSP

from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs

from matplotlib.pyplot import cm



def main():

    # Symbols of Countries
    coun_names = pd.read_csv('./country_names.csv')
    country_names = list(coun_names['Country'])
    

    df_data = pd.read_csv('./data_2022' + '.csv', index_col='Country')
    types = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1])
    matrix = df_data.to_numpy()
    average = np.mean(matrix, axis = 0)
    weights = mcda_weights.critic_weighting(matrix)

    list_corrs = []
    coeffs = np.arange(0.0, 0.55, 0.05)
    
    df_results = pd.DataFrame(index = df_data.index)
    df_rank_v = pd.DataFrame(index = df_data.index)

    # First simulation
    for coeff in coeffs:

        s_coeff = np.ones(matrix.shape[1]) * coeff

        spotis = SPOTIS_SSP()
        pref = spotis(matrix, weights, types, s_coeff = s_coeff)
        rank_spotis = rank_preferences(pref, reverse = False)
        
        # SSP-VIKOR
        ssp_vikor = SSP_VIKOR()

        pref_ssp_vikor = ssp_vikor(matrix, weights, types, s_coeff = s_coeff)
        rank_ssp_vikor = rank_preferences(pref_ssp_vikor, reverse = False)
        df_results[str(np.round(coeff, 2))] = rank_ssp_vikor

        df_rank_v[str(coeff)] = rank_ssp_vikor

        list_corrs.append(corrs.weighted_spearman(rank_spotis, rank_ssp_vikor))

    fig, ax = plt.subplots(figsize=(9, 5))
    plt.plot(coeffs, list_corrs, linewidth = 3)
    plt.xlabel(r'$s$' + ' coefficient in SSP-VIKOR and SPOTIS', fontsize = 14)
    plt.ylabel(r'$r_w$' + ' correlation', fontsize = 14)
    ax.tick_params(axis = 'both', labelsize = 14)
    
    plt.grid(True, linestyle = '-.')
    plt.tight_layout()
    plt.savefig('./results/ssp_vikor_spotis1.pdf')
    plt.show()


    # ==================================================================================
    # plot figure with sensitivity analysis

    plt.figure(figsize = (9, 6))
    for k in range(df_rank_v.shape[0]):
        plt.plot(coeffs, df_rank_v.iloc[k, :], '.-')

        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        
        plt.annotate(country_names[k], (x_max, df_rank_v.iloc[k, -1]),
                        fontsize = 12, style='italic',
                        horizontalalignment='left')

    plt.xlabel(r'$s$' + ' coefficient value', fontsize = 12)
    plt.ylabel("Rank", fontsize = 12)
    
    plt.xticks(coeffs, fontsize = 12)
    plt.yticks(ticks=np.arange(1, len(country_names) + 1, 1), fontsize = 12)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle = 'dashdot')
    plt.title('All criteria compensation reduction', fontsize = 12)
    plt.tight_layout()
    plt.savefig('results/sust_coeff' + '.pdf')
    plt.show()


    # different colors of lines on chart
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Dark2(i))
    for i in range(10):
        color.append(cm.tab20b(i))
    

    df_results = df_results.rename_axis('Country')
    print(df_results)
    df_results.to_csv('./results/df_rankings_ssp_vikor.csv')

    # ===================================================================================
    # Comparison of SSP-VIKOR & SPOTIS
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, coeff_s in enumerate(coeffs):
        s = np.ones(matrix.shape[1]) * coeff_s
        list_corrs = []
        for coeff_v in coeffs:
            
            spotis = SPOTIS_SSP()
            pref = spotis(matrix, weights, types, s_coeff = s)
            rank_spotis = rank_preferences(pref, reverse = False)
            
            # SSP-VIKOR
            ssp_vikor = SSP_VIKOR()

            s_coeff = np.ones(matrix.shape[1]) * coeff_v
            pref_ssp_vikor = ssp_vikor(matrix, weights, types, s_coeff = s_coeff)
            rank_ssp_vikor = rank_preferences(pref_ssp_vikor, reverse = False)

            list_corrs.append(corrs.weighted_spearman(rank_spotis, rank_ssp_vikor))
        plt.plot(coeffs, list_corrs, linewidth = 3, color = color[i], label = str(np.round(coeff_s, 2)))

    plt.legend(bbox_to_anchor=(1.01, 1),
                         loc='upper left', borderaxespad=0., title = 'Bounds compensation\n reduction in SPOTIS', fontsize = 14)
    plt.xlabel(r'$s$' + ' coefficient in SSP-VIKOR', fontsize = 14)
    plt.ylabel(r'$r_w$' + ' correlation', fontsize = 14)
    ax.tick_params(axis = 'both', labelsize = 14)
    
    plt.grid(True, linestyle = '-.')
    plt.tight_layout()
    plt.savefig('./results/ssp_vikor_spotis2.pdf')
    plt.show()


    # ========================================================================================
    # Simulation benchmarking
    coeffs = np.arange(0.0, 1.1, 0.1)

    df = pd.DataFrame()
    x = []
    y = []

    for i in range(0, 10000):

        matrix = np.random.uniform(1, 1000, size = (20, 10))
        types = np.ones(10)
        weights = mcda_weights.critic_weighting(matrix)
        average = np.mean(matrix, axis = 0)


        for coeff in coeffs:
            
            s_coeff = np.ones(matrix.shape[1]) * coeff
            spotis = SPOTIS_SSP()
            
            pref = spotis(matrix, weights, types, s_coeff = s_coeff)
            rank_spotis = rank_preferences(pref, reverse = False)
            
            # SSP-VIKOR
            ssp_vikor = SSP_VIKOR()
            
            pref_ssp_vikor = ssp_vikor(matrix, weights, types, s_coeff = s_coeff)
            rank_ssp_vikor = rank_preferences(pref_ssp_vikor, reverse = False)

            x.append(np.round(coeff, 2))
            y.append(corrs.weighted_spearman(rank_spotis, rank_ssp_vikor))

    df['coeff'] = x
    df['corr'] = y

    # plot violin
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.violinplot(data=df, x = 'coeff', y = 'corr', fill=False)
    ax.grid(True, linestyle = '-.')
    plt.yticks(fontsize = 14)
    plt.xticks(fontsize = 14)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'$s$' + ' coefficient', fontsize = 14)
    ax.set_ylabel(r'$r_w$' + ' correlation', fontsize = 14)
    plt.title('Comparison of SSP-VIKOR & SPOTIS rankings')
    plt.tight_layout()
    plt.savefig('./results/ssp_vikor_spotis_bench.pdf')
    plt.show()
    
    
if __name__ == '__main__':
    main()