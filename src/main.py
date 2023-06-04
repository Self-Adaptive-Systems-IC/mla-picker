#!/usr/bin/env python

import sys
import pickle
import numpy as np
from pycaret import classification
from tabulate import tabulate
import statistics
import time
import pandas as pd

from pycaret.classification import *

from utils.cross_validation_utils import cross_validation
from utils.find_best_agm_utils import find_best_agm
from utils.teste_tukey_utils import tukey_test
from utils.grid_search_utils import config_selector

from utils.neural_network_linear_regression_utils_1 import foo
# from utils.random_search_utils import config_selector
agms = {0:'tree', 1:'random_forest', 2:'svc'}

# Define pickle file squeme
def read_file_pkl(file: str):
    with open(file, 'rb') as f:
        # X_train, X_test, y_train, y_test = pickle.load(f)
        X_train, y_train, X_test, y_test = pickle.load(f)
    X_data = np.concatenate((X_train, X_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    return X_data, y_data

# Read a csv file and create a dataframe
def read_file_csv(file: str):
    data = pd.read_csv(file, index_col=None)
    return data

def print_table(results: list, agms: dict):
    result_t = []
    for i, array in enumerate(results):
        m = statistics.fmean(array)
        result_t.append([agms[i], m])
    print("***************Table***************")
    print(tabulate(result_t, headers=['Agm','Mean']))
    print("***********************************")
    
def menu():
    print("--- MENU ---")
    print("0: Melhor algoritmo")
    print("1: Tabela com as médias")
    option = int(input("Select: "))
    return option

foo_helper = {
    0: find_best_agm,
    1: print_table
}

def main(file_path: str):
    # start = time.time()
    # print(f"Running for the dataset {file_path}")
    # # Get X and y values from file
    # X_data, y_data = read_file_pkl(file_path)
    # config_selector(X_data, y_data)
    # # Exacute cross validation between agms
    # results = cross_validation(X_data, y_data)
    # # Using tukey test to find the best agm
    # tukey_test(results, agms)
    # end = time.time()
    # print_table(results, agms)
    # print(f"\nElapsed {end-start}s")
    # # op = menu()
    # # foo_helper[op](results, agms)
    # # find_best_agm(results, agms)
    target_position = -1

    data = pd.read_csv(file_path)

    target: str = str(data.columns[target_position])

    s = ClassificationExperiment()
    s = setup(
            data=data, # Dataset
            target=target, # Target Column
            train_size=0.8, # Proportion of the dataset to be used to training and validation
            numeric_imputation='knn', # Strategy for numerical columns (at missing values)
            categorical_imputation='mode', # Strategy for categorical columns (at missing values)
            html=False, # Prevents runtime display at command line
            verbose=False,
            fold_strategy='kfold',
            fold=2,
            session_id=None, # Controls the randomness of experiment
            n_jobs=-1 # How much jobs will running paralel
            )

    # Select the best model
    best_model = s.compare_models(
            turbo=False,
            sort='Recall',
            #fold=5
            )

    # Tunning the best model
    tunned_model = s.tune_model(
            best_model,
            n_iter=50,
            search_library='scikit-learn',
            search_algorithm='random',
            optimize='Recall'
            )

    print(tunned_model)
    
    foo(s.X_train,s.y_train, s.X_test, s.y_test)



if __name__ == "__main__":
    try:
        file_path = sys.argv[1]
    except IndexError:
        print("Please inform the file path")
        sys.exit(1)
    main(file_path)
