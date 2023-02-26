import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
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
    
    print("\n", test_tukey, "\n")
    # Create a data frame with th columns reject1, reject2 and total_sum
    tukey_data = handle_tukey_result(test_tukey)
    agms_compare = find_agms_to_compare(tukey_data)
    
    best_results = []
    agms2 = {}
    i = 0
    for element in agms_compare:
        key=list(agms.keys())[list(agms.values()).index(element)]
        best_results.append(results[key])
        
        agms2[i] = element
        i += 1
    
    find_best_agm(best_results, agms2)  
    # Get the max value of column total_sum
    # max_value = tukey_data['total_sum'].max()
    # Get the line who have the max values of total_sum
    # ids = tukey_data[tukey_data.iloc[:, 2] == max_value].index
    # Find the max accuracy
    # best_results = []
    # for element in ids:
    #     key=list(agms.keys())[list(agms.values()).index(element)]
    #     best_results.append(results[key])
    # find_best_agm(best_results, agms)
    
def handle_tukey_result(test_tukey):    
    values = test_tukey._results_table.data[1:]
    
    filter = []
    for element in values:
        filter.append([element[0], element[1], element[-1]])
    
    tukey_data = pd.DataFrame(data=filter, columns=['group1', 'group2', 'reject'])

    return filter

def find_agms_to_compare(tukey_data):
    true_index = []
    false_index = []
    agm_to_compare = []
    agm_not_insert = []
    
    for e in tukey_data:
        # print(e)
        r_value = e[-1]
        if r_value:
            true_index.append(e)
        else:
            false_index.append(e)
    
    for e in false_index:
        if not e[0] in agm_to_compare:
            agm_to_compare.append(e[0])
            agm_not_insert.append(e[1])
        elif not e[1] in agm_to_compare:
            agm_to_compare.append(e[1])
            agm_not_insert.append(e[0])
    
    for e in true_index:
        if not e[0] in agm_to_compare:
            if not e[0] in agm_not_insert:
                agm_to_compare.append(e[0])
        elif not e[1] in agm_to_compare:
            if not e[1] in agm_not_insert:
                agm_to_compare.append(e[1])
    return agm_to_compare