import os
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm

from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import correlations as corrs
from daria import DARIA

from pyfdm.methods import fTOPSIS, fVIKOR, fEDAS
from pyfdm.weights import standard_deviation_weights
from pyfdm.methods.fuzzy_sets import tfn


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value



def plot_barplot(df_plot, x_name, y_name, title):
    """
    Display stacked column chart of weights for criteria for `x_name == Weighting methods`
    and column chart of ranks for alternatives `x_name == Alternatives`

    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights calculated different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of chart title, Weighting methods or Criteria

    Examples
    ----------
    >>> plot_barplot(df_plot, x_name, y_name, title)
    """
    
    list_rank = np.arange(1, len(df_plot) + 2, 2)
    stacked = True
    width = 0.6
    if x_name == 'Alternatives':
        stacked = False
        width = 0.8
        ncol = 2
    else:
        ncol = 5
    
    ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (10,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    if x_name == 'Alternatives':
        ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 11)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./results/bar_chart_' + title[-4:] + '.pdf', bbox_inches = 'tight')
    plt.show()



# heat maps with correlations
def draw_heatmap(df_new_heatmap, title):
    """
    Visualization method to display heatmap with correlations of compared rankings generated using different methods
    
    Parameters
    ----------
        data : DataFrame
            DataFrame with correlation values between compared rankings
        title : str
            title of chart containing name of used correlation coefficient
    Examples
    ---------
    >>> draw_heatmap(df_new_heatmap, title)
    """
    plt.figure(figsize = (10, 8))
    sns.set(font_scale = 1.7)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="PRGn",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Methods')
    plt.title('Correlation: ' + title)
    plt.tight_layout()
    title = title.replace("$", "")
    plt.savefig('./results/' + 'correlations_' + title + '.pdf', bbox_inches = 'tight')
    plt.show()


def main():
    
    path = './dataset'
    # Number of countries
    m = 27

    # Symbols of Countries
    coun_names = pd.read_csv('./dataset/country_symbols.csv')
    country_names = list(coun_names['Symbol'])

    str_years = [str(y) for y in range(2015, 2021)]

    '''
    # preparing fuzzy data sets (input)
    for el, year in enumerate(str_years):
        file = 'data_sdg11_' + str(year) + '_fuzzy' + '.csv'
        pathfile = os.path.join(path, file)

        # crisp decision matrix for given year
        data = pd.read_csv(pathfile, index_col = 'Country')

        # new fuzzy decision matrix with triangular fuzzy values (low, medium, high)
        fuzzy_df = pd.DataFrame(index = country_names)

        for column in data:

            tab = np.array(data[column])
            print(tab)
            print('---------------------------------------')

            fuzzy_df[column + ' low'] = tab - 0.2 * tab
            fuzzy_df[column] = tab
            fuzzy_df[column + ' high'] = tab + 0.2 * tab

        fuzzy_df = fuzzy_df.rename_axis('Country')
        fuzzy_df.to_csv('./dataset/' + 'data_sdg11_' + str(year) + '_fuzzy' + '.csv')
        '''


    # dataframe for annual results fuzzy TOPSIS
    preferences_t = pd.DataFrame(index = country_names)
    rankings_t = pd.DataFrame(index = country_names)

    # dataframe for annual results fuzzy VIKOR
    preferences_v = pd.DataFrame(index = country_names)
    rankings_v = pd.DataFrame(index = country_names)

    # dataframe for annual results fuzzy EDAS
    preferences_e = pd.DataFrame(index = country_names)
    rankings_e = pd.DataFrame(index = country_names)

    mat_avg = np.zeros((27, 9, 3))
    # initialization of F-TOPSIS
    f_topsis = fTOPSIS(normalization=tfn.normalizations.minmax_normalization)
    # initialization of F-VIKOR
    f_vikor = fVIKOR()
    # initialization of F-EDAS
    f_edas = fEDAS()

    # dataframes for results summary
    pref_summary = pd.DataFrame(index = country_names)
    rank_summary = pd.DataFrame(index = country_names)

    for el, year in enumerate(str_years):
        file = 'data_sdg11_' + str(year) + '_fuzzy' + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Country')
        
        # types: 1 profit -1 cost
        types = np.array([-1, -1, 1, -1, -1, 1, 1, 1, -1])
        
        list_of_cols = list(data.columns)
        # matrix
        pre_matrix = data.to_numpy()

        # Block size - shape of fuzzy number, for example 3 for triangular fuzzy number, 4 for trapezoidal fuzzy number
        block_size = 3

        # Transforming an array into an array with nested fuzzy numbers with shape specified by block_size
        matrix = np.array([pre_matrix[i, j:j + block_size] for i in range(pre_matrix.shape[0]) for j in range(0, pre_matrix.shape[1], block_size)])
        matrix = matrix.reshape(pre_matrix.shape[0], -1, block_size)
        

        # fuzzy weights
        weights = standard_deviation_weights(matrix)

        # F-TOPSIS annual
        pref_t = f_topsis(matrix, weights, types)
        
        rank_t = f_topsis.rank()
        
        preferences_t[year] = pref_t
        rankings_t[year] = rank_t

        # F-VIKOR annual
        pref_v = f_vikor(matrix, weights, types)[2]
        rank_v = f_vikor.rank()[2]
        

        preferences_v[year] = pref_v
        rankings_v[year] = rank_v

        # F-EDAS annual
        pref_e = f_edas(matrix, weights, types)
        rank_e = f_edas.rank()

        preferences_e[year] = pref_e
        rankings_e[year] = rank_e

        mat_avg += matrix



    preferences_t.to_csv('./results/fuzzy TOPSIS annual utility function values.csv')
    preferences_v.to_csv('./results/fuzzy VIKOR annual utility function values.csv')
    preferences_e.to_csv('./results/fuzzy EDAS annual utility function values.csv')
    
    rankings_t.to_csv('./results/Fuzzy TOPSIS annual rankings.csv')
    rankings_v.to_csv('./results/Fuzzy VIKOR annual rankings.csv')
    rankings_e.to_csv('./results/Fuzzy EDAS annual rankings.csv')
    # Create dataframes for summary of preferences and rankings
    
    # AVG performances of alternatives
    mat_avg = mat_avg / len(str_years)
    # weights Entropy
    weights_avg = standard_deviation_weights(mat_avg)

    # F-TOPSIS AVG
    pref_avg_t = f_topsis(mat_avg, weights_avg, types)
    # rank_avg_t = rank_preferences(pref_avg_t, reverse=True)
    rank_avg_t = f_topsis.rank()

    # F-VIKOR AVG
    pref_avg_v = f_vikor(mat_avg, weights_avg, types)[2]
    # rank_avg_v = rank_preferences(pref_avg_v, reverse=False)
    rank_avg_v = f_vikor.rank()[2]

    # F-EDAS AVG
    pref_avg_e = f_edas(mat_avg, weights_avg, types)
    # rank_avg_t = rank_preferences(pref_avg_t, reverse=True)
    rank_avg_e = f_edas.rank()

    pref_summary['F-TOPSIS'] = pref_avg_t
    rank_summary['F-TOPSIS'] = rank_avg_t

    pref_summary['F-VIKOR'] = pref_avg_v
    rank_summary['F-VIKOR'] = rank_avg_v

    pref_summary['F-EDAS'] = pref_avg_e
    rank_summary['F-EDAS'] = rank_avg_e

    # saved temporal matrix with annual results
    
    # PLOT  TOPSIS =======================================================================
    # annual rankings chart
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Set2(i))
    for i in range(10):
        color.append(cm.tab10(i))
    for i in range(8):
        color.append(cm.Pastel2(i))
    
    # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": "-"})
    ticks = np.arange(1, m + 2, 2)

    x1 = np.arange(0, len(str_years))

    plt.figure(figsize = (8, 7))
    for i in range(rankings_t.shape[0]):
        c = color[i]
        plt.plot(x1, rankings_t.iloc[i, :], 'o-', color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(country_names[i], (x_max - 0.15, rankings_t.iloc[i, -1]),
                        fontsize = 14, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Year", fontsize = 14)
    plt.ylabel("Rank", fontsize = 14)
    plt.xticks(x1, str_years, fontsize = 14)
    plt.yticks(ticks, fontsize = 14)
    plt.xlim(x_min - 0.2, x_max + 0.2)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('F-TOPSIS Rankings', fontsize = 14)
    plt.tight_layout()
    plt.savefig('./results/rankings_annual_fuzzy_TOPSIS' + '.pdf', bbox_inches = 'tight')
    plt.show()
    
    # PLOT  VIKOR =======================================================================
    # annual rankings chart
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Set2(i))
    for i in range(10):
        color.append(cm.tab10(i))
    for i in range(8):
        color.append(cm.Pastel2(i))
    
    # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": "-"})
    ticks = np.arange(1, m + 2, 2)

    x1 = np.arange(0, len(str_years))

    plt.figure(figsize = (8, 7))
    for i in range(rankings_v.shape[0]):
        c = color[i]
        plt.plot(x1, rankings_v.iloc[i, :], 'o-', color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(country_names[i], (x_max - 0.15, rankings_v.iloc[i, -1]),
                        fontsize = 14, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Year", fontsize = 14)
    plt.ylabel("Rank", fontsize = 14)
    plt.xticks(x1, str_years, fontsize = 14)
    plt.yticks(ticks, fontsize = 14)
    plt.xlim(x_min - 0.2, x_max + 0.2)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('F-VIKOR Rankings', fontsize = 14)
    plt.tight_layout()
    plt.savefig('./results/rankings_annual_fuzzy_VIKOR' + '.pdf', bbox_inches = 'tight')
    plt.show()

    # PLOT  EDAS =======================================================================
    # annual rankings chart
    color = []
    for i in range(9):
        color.append(cm.Set1(i))
    for i in range(8):
        color.append(cm.Set2(i))
    for i in range(10):
        color.append(cm.tab10(i))
    for i in range(8):
        color.append(cm.Pastel2(i))
    
    # sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": "-"})
    ticks = np.arange(1, m + 2, 2)

    x1 = np.arange(0, len(str_years))

    plt.figure(figsize = (8, 7))
    for i in range(rankings_e.shape[0]):
        c = color[i]
        plt.plot(x1, rankings_e.iloc[i, :], 'o-', color = c, linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(country_names[i], (x_max - 0.15, rankings_e.iloc[i, -1]),
                        fontsize = 14, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Year", fontsize = 14)
    plt.ylabel("Rank", fontsize = 14)
    plt.xticks(x1, str_years, fontsize = 14)
    plt.yticks(ticks, fontsize = 14)
    plt.xlim(x_min - 0.2, x_max + 0.2)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('F-EDAS Rankings', fontsize = 14)
    plt.tight_layout()
    plt.savefig('./results/rankings_annual_fuzzy_EDAS' + '.pdf', bbox_inches = 'tight')
    plt.show()

    

    # ======================================================================
    # fuzzy DARIA-TOPSIS - 1
    # ======================================================================
    # DARIA (DAta vaRIAbility) temporal approach
    # preferences includes preferences of alternatives for evaluated years
    df_varia_fin = pd.DataFrame(index = country_names)
    df = preferences_t.T
    matrix = df.to_numpy()

    # TOPSIS orders preferences in descending order
    met = 'topsis'
    type = 1

    # calculate efficiencies variability using DARIA methodology
    daria = DARIA()
    # calculate variability values with Gini coefficient
    var = daria._gini(matrix)
    # calculate variability directions
    dir_list, dir_class = daria._direction(matrix, type)

    # actions for next stage of research
    df_varia_fin[met.upper()] = list(var)
    df_varia_fin[met.upper() + ' dir'] = list(dir_class)

    df_results = pd.DataFrame(index = df.columns)
    df_results['Variability'] = list(var)
    
    # list of directions
    df_results['Direction'] = dir_list

    # final calculation
    # data with alternatives' rankings' variability values calculated with Gini coeff and directions
    G_df = copy.deepcopy(df_varia_fin)

    # data with alternatives' efficiency of performance calculated for the recent period
    S_df = copy.deepcopy(preferences_t)

    # ==============================================================
    S = S_df['2020'].to_numpy()

    G = G_df[met.upper()].to_numpy()
    dir = G_df[met.upper() + ' dir'].to_numpy()

    # update efficiencies using DARIA methodology
    # final updated preferences (utility function values)
    final_S = daria._update_efficiency(S, G, dir)

    # TOPSIS has descending ranking from prefs
    rank = rank_preferences(final_S, reverse = True)

    df_results['Fuzzy DARIA-TOPSIS pref'] = final_S
    df_results['Fuzzy DARIA-TOPSIS rank'] = rank
    df_results = df_results.rename_axis('Country')
    df_results.to_csv('./results/final temporal results FUZZY DARIA-TOPSIS.csv')

    pref_summary['F-DARIA-TOPSIS'] = final_S
    rank_summary['F-DARIA-TOPSIS'] = rank


    # =====================================================================
    # saving whole results
    pref_summary = pref_summary.rename_axis('Country')
    rank_summary = rank_summary.rename_axis('Country')
    pref_summary.to_csv('./results/utility function values summary.csv')
    rank_summary.to_csv('./results/rankings summary.csv')


    # plot
    # correlations for PLOT
    method_types = list(rank_summary.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    dict_new_heatmap_rs = copy.deepcopy(dict_new_heatmap_rw)

    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            dict_new_heatmap_rw[j].append(corrs.weighted_spearman(rank_summary[i], rank_summary[j]))
            dict_new_heatmap_rs[j].append(corrs.spearman(rank_summary[i], rank_summary[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_rs = pd.DataFrame(dict_new_heatmap_rs, index = method_types[::-1])
    df_new_heatmap_rs.columns = method_types

    # correlation matrix with rw coefficient
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')

    # correlation matrix with rs coefficient
    draw_heatmap(df_new_heatmap_rs, r'$r_s$')
    

if __name__ == '__main__':
    main()