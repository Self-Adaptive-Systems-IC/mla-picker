# Importing algorithm for tunning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# Importing machine learning models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Importing utils libs
# import pickle
import yaml
# import numpy as np
from datetime import datetime
from tabulate import tabulate

# Dict to use like a switch to select de machine learning model
models = {
        'svc': SVC,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        }

# Function to read a yaml file with hyperparameters definition and convert to dict
def import_hyperparameters():
    with open('./src/config/config_models.yml') as f:
        dict = yaml.safe_load(f)
    return dict

# Function to execute the machine learning tunning
def hyperparameters_selector(X_data, y_data, op=0, cv_values=2, n_iter=20, n_jobs=2):
    """
    Function to execute the machine learning tunning using grid search or randomized search.
    Return the best hyperparameters selection to each machine learning model, pre seted

    Parameters
    ----------
        X_data : array
            The data to fit. Can be a list or an array.        
        y_data : array
            The target variable to try predict in the case of supervised learning.
        op : int
            Define the tunning method. 0: Randomized Search and 1: Grid Search
        cv_values : int
            Determines the cross-validation splitting strategy.
        n_iter : int
            Number of parameter settings that are sampled. n_iter trades of runtime vs quality of the solution.
        n_jobs : int or None
            Number of jobs to run in parallel. None means 1 and -1 means using all processors.
    """
    configs = {} # config dict with the best params
    hyperparameters = import_hyperparameters() # get the hyperparameters from yaml file
    results = [] # array to save all results, for printing
    
    for keys, values in models.items():
        # Select the algorithm for tunning
        if op == 0:
            search = RandomizedSearchCV(
                    estimator=values(), # Set the machine learning model
                    param_distributions=hyperparameters[keys], # Dictionary with parameters
                    scoring='accuracy', # Strategy to evaluate the performance
                    cv=cv_values, # Set the cross-validation splitting strategy
                    n_iter=n_iter, # Number of parameter settings that are sampled
                    verbose=3, # Controls the verbosity (how much message will be show)
                    n_jobs=n_jobs # Number os jobs to run in parallel
                    )
        else:
            search = GridSearchCV(
                    estimator=values(), # Set the machine learning model
                    param_grid=hyperparameters[keys], # Dictionary with parameters
                    scoring='accuracy', # Strategy to evaluate the performance
                    cv=cv_values, # Set the cross-validation splitting strategy
                    verbose=3, # Controls the verbosity (how much message will be show)
                    n_jobs=n_jobs # Number of jobs to tun in parallel
                    )
        print(f'START - {keys}')
        start = datetime.now() # Start the fit
        search.fit(X_data, y_data) # Training data with all set of params 
        end = datetime.now() # End the fit
        print(f'END - {keys}')
        params = search.best_params_ # Get the best param
        score = search.best_score_ # Get the best score
        # print(keys)
        configs.update({keys: params}) # Save the best param
        results.append([score, (end-start), params]) # Save the score, time and best param, for printing
    # Print the results
    print(tabulate(results, headers=['Socre', 'Time', 'Result']))
    return configs # Return de best params for each model
