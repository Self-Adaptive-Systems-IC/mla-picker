from scipy.stats import f_oneway
import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
import statistics
from utils.find_best_agm_utils import find_best_agm
def tukey_test(results, agms):
    alpha = 0.05
    _, p = f_oneway(results[0], results[1])
    if p > alpha:
        print('Hipótese alternativa rejeitada. Resultados são iguais')
        return
    size_r = len(results[0])
    results_dict = {
        'accuracy': np.concatenate(results),
        'agm': np.concatenate([
            np.full(shape=size_r, fill_value='tree'),
            np.full(shape=size_r, fill_value='random_forest'),
        ])
    }
    # print(results_dict)
    results_df = pd.DataFrame(results_dict)
    
    comp_agm = MultiComparison(results_df['accuracy'], results_df['agm'])
    test_tukey = comp_agm.tukeyhsd()
    
    # print(test_tukey)
    reject = test_tukey.reject[0]
    if reject:
        find_best_agm(results, agms)
    else:
        print("Ambos métodos possuem mesmo resultado.")
    return 0
