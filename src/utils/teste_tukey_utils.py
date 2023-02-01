import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
from utils.find_best_agm_utils import find_best_agm

def tukey_test(results, agms):
    alpha = 0.05
    _, p = f_oneway(results[0], results[1])
    # Verify if p value is greatest the alpha value
    ## True: stop executing, because all values are equal
    ## False: keep executing, because all values are deferents
    if p > alpha:
        print('Hipótese alternativa rejeitada. Resultados são iguais')
        return
    size_r = len(results[0])
    results_dict = {
        'accuracy': np.concatenate(results),
        'agm': np.concatenate([
            np.full(shape=size_r, fill_value='tree'),
            np.full(shape=size_r, fill_value='random_forest'),
            np.full(shape=size_r, fill_value='svc'),
        ])
    }
    results_df = pd.DataFrame(results_dict)
    # Execute tukey test to compare elements pair to pair
    comp_agm = MultiComparison(results_df['accuracy'], results_df['agm'])
    test_tukey = comp_agm.tukeyhsd()
    # Create a data frame with th columns reject1, reject2 and total_sum
    tukey_data = handle_tukey_result(test_tukey)
    # Get the max value of column total_sum
    max_value = tukey_data['total_sum'].max()
    # Get the line who have the max values of total_sum
    ids = tukey_data[tukey_data.iloc[:, 2] == max_value].index
    # Find the max accuracy
    best_results = []
    for element in ids:
        key=list(agms.keys())[list(agms.values()).index(element)]
        best_results.append(results[key])
    find_best_agm(best_results, agms)
    
def handle_tukey_result(test_tukey):
    tukey_data = pd.DataFrame(data=test_tukey._results_table.data[1:], columns = test_tukey._results_table.data[0])
    group1_comp =tukey_data.loc[tukey_data.reject == True].groupby('group1').reject.count()
    group2_comp = tukey_data.loc[tukey_data.reject == True].groupby('group2').reject.count()
    tukey_data = pd.concat([group1_comp, group2_comp], axis=1)
    tukey_data = tukey_data.fillna(0)
    tukey_data.columns = ['reject1', 'reject2']
    tukey_data['total_sum'] = tukey_data.reject1 + tukey_data.reject2
    return tukey_data