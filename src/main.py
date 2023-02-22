#!/usr/bin/env python

import sys
import pickle
import numpy as np
from tabulate import tabulate
import statistics

from utils.cross_validation_utils import cross_validation
from utils.find_best_agm_utils import find_best_agm
from utils.teste_tukey_utils import tukey_test

agms = {0:'tree', 1:'random_forest', 2:'svc'}

# Define pickle file squeme
def read_file(file: str):
    with open(file, 'rb') as f:
        # X_train, X_test, y_train, y_test = pickle.load(f)
        X_train, y_train, X_test, y_test = pickle.load(f)
    X_data = np.concatenate((X_train, X_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    return X_data, y_data

def print_table(results: list, agms: dict):
    result_t = []
    for i, array in enumerate(results):
        m = statistics.fmean(array)
        result_t.append([agms[i], m])
    print("***************Table***************")
    print(tabulate(result_t, headers=['Agm','Mean']))
    print("***********************************")
    
def menu():
    print("\n\033[1m--- MENU ---\033[0m")
    print("0: Melhor algoritmo")
    print("1: Tabela com as médias")
    option = int(input("Select: "))
    return option

foo_helper = {
    0: find_best_agm,
    1: print_table
}

def main(file_path: str):
    print(f"Running for the dataset {file_path}")
    # Get X and y values from file
    X_data, y_data = read_file(file_path)
    # Exacute cross validation between agms
    results = cross_validation(X_data, y_data)
    # Using tukey test to find the best agm
    tukey_test(results, agms)
    # print_table(results, agms)
    # op = menu()
    # foo_helper[op](results, agms)
    # find_best_agm(results, agms)

if __name__ == "__main__":
    try:
        file_path = sys.argv[1]
    except IndexError:
        print("Please inform the file path")
        sys.exit(1)
    main(file_path)
