import statistics

def find_best_agm(results, agms):
    """
    Function to find the algorithm with the best accuracy

    Parameters
    ----------
        results : array
            The array of array with the accuracy for each algorithm.        
        agms : dict
            The dict to find the name of each algorithm.
    """
    max = { 'agm': None, 'value': 0 }   
    for i, array in enumerate(results):
        m = statistics.fmean(array)
        if m > max['value']:
            max['agm'] = agms[i]
            max['value'] = m
    # Print results
    print(f"\nBest agm is \033[1m{max['agm'].replace('_', ' ')}\033[0m")
    print(f"Mean equal to \033[1m{max['value']:.4}\033[0m\n")