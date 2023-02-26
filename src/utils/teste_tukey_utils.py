import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
from utils.find_best_agm_utils import find_best_agm

# Main function that execute tukey test to find the best agm
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
    tukey_data = filter_tukey_result(test_tukey)
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

# Filter the values is need from tukey test
def filter_tukey_result(test_tukey):    
    values = test_tukey._results_table.data[1:]
    
    filter = []
    for element in values:
        filter.append([element[0], element[1], element[-1]])
    
    tukey_data = pd.DataFrame(data=filter, columns=['group1', 'group2', 'reject'])

    return filter

# Get the agms that reject null hypothesis
def find_agms_to_compare(tukey_data):
    group = [[], []]
    agm_to_compare = []
    agm_not_insert = []
    
    for e in tukey_data:
        r_value = e[-1]
        if r_value:
            group[0].append(e)
        else:
            group[1].append(e)
    
    for e in group[1]:
        if not e[0] in agm_to_compare:
            agm_to_compare.append(e[0])
            agm_not_insert.append(e[1])
        elif not e[1] in agm_to_compare:
            agm_to_compare.append(e[1])
            agm_not_insert.append(e[0])
    
    for e in group[0]:
        if not e[0] in agm_to_compare and not e[0] in agm_not_insert:
                agm_to_compare.append(e[0])
        elif not e[1] in agm_to_compare and not e[1] in agm_not_insert:
                agm_to_compare.append(e[1])
    return agm_to_compare
