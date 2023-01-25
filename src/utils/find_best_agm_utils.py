import statistics

def find_best_agm(results, agms):
    max = { 'agm': None, 'value': 0 }   
    for i, array in enumerate(results):
        m = statistics.fmean(array)
        if m > max['value']:
            max['agm'] = agms[i]
            max['value'] = m
    print(f"Best agm is {max['agm']} whith mean equal to {max['value']}")