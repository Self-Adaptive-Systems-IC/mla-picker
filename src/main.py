#!/usr/bin/env python

import sys
import pickle
import numpy as np
from pycaret import classification
from tabulate import tabulate
import statistics
# import time
import pandas as pd
import os
from datetime import date#, time

from pycaret.classification import *

from utils.cross_validation_utils import cross_validation
from utils.find_best_agm_utils import find_best_agm
from utils.teste_tukey_utils import tukey_test
from utils.grid_search_utils import config_selector

from utils.neural_network_earlystopping_training_utils import early_stopping
from sklearn.metrics import accuracy_score, recall_score
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

def save_feature_importance(setup, model):
    print(os.path.abspath(os.getcwd()))
    os.chdir('./src')
    print(os.listdir())
    print(os.path.abspath(os.getcwd()))
    os.chdir('./data')
    print(os.path.abspath(os.getcwd()))
    os.chdir('./images')
    print(os.path.abspath(os.getcwd()))  
    setup.plot_model(model, plot = 'feature', save=True)
    os.chdir('../../..')
    
def save_model(setup, model):
    os.chdir('./src/data/saved_models')
    # setup.save_model(model, 'best_model' + '_' + str(date.today()))
    setup.save_model(model, 'model')
    os.chdir('../../..')
    
def save_api(setup, model, file_name=''):
    name = 'api_'
    if file_name == '':
        name = name+str(date.today())
    else:
        name = name+file_name
    setup.create_api(model, name)
    # setup.create_docker(name) # To create a docker file
    # Python program to replace text in a file
    a1 = 'return {"prediction": predictions["prediction_label"].iloc[0]}'
    a2 = 'return {"prediction": predictions["prediction_label"].iloc[0], "score": predictions["prediction_score"].iloc[0]}'
    b1 = 'prediction=0'
    c1 = 'prediction=1'
    b2 = '**{"prediction":0, "score":0.8}'
    
    # Read in the file
    with open(name+'.py', 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace(a1, a2)
    filedata = filedata.replace(b1, b2)
    filedata = filedata.replace(c1, b2)

    # Write the file out again
    with open(name+'.py', 'w') as file:
        file.write(filedata)

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

    file_name = (file_path.split("/")[-1].split('.')[0])


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
            fold=3,
            session_id=None, # Controls the randomness of experiment
            n_jobs=4 # How much jobs will running paralel
            )

    # Select the best model
    best_model = s.compare_models(
            turbo=False,
            sort='Recall',
            #fold=5
            exclude=['dt']
            )

    # Tunning the best model
    tunned_model = s.tune_model(
            best_model,
            n_iter=5,
            search_library='scikit-learn',
            search_algorithm='random',
            optimize='Recall'
            )

    print(tunned_model)
    

    model_early_stopping = early_stopping(s.X_train,s.y_train, s.X_test, s.y_test)
    
    print('Compare and select between the early_stopping and pycaret model -------------------------------------------')
    selected_model = s.compare_models(include=[tunned_model, model_early_stopping], sort="Accuracy")
    print('-----------------------------------------------------------------------------------------------------------')

    save_api(s, selected_model, file_name)
    
    # save_model(s, selected_model)

    # s.create_app(selected_model)

    # save_feature_importance(s, )
    # save_feature_importance(s, model_early_stopping)



if __name__ == "__main__":
    try:
        file_path = sys.argv[1]
    except IndexError:
        print("Please inform the file path")
        sys.exit(1)
    main(file_path)
