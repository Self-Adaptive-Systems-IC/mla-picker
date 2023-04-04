import yaml
import pickle
import numpy as np
# from sklearn.model_selection import RandomizedSearch
from sklearn.model_selection import RandomizedSearchCV
from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from utils.Loader import Loader

models = {
        'svc': SVC,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier
        }

def import_hyperparameters():
    with open('./src/config/config_models.yml') as f:
        dict = yaml.safe_load(f)
    return dict
   
def config_selector(X_data, y_data):
    hyperparameters = import_hyperparameters()
    results = []
    cv_values = 3

    for keys, values in models.items():
        r_search = RandomizedSearchCV(
                estimator=values(),
                param_distributions=hyperparameters[keys],
                scoring='accuracy',
                cv=cv_values,
                n_iter=50,
                verbose=1,
                n_jobs=-1
                )
        print(f'Start - {keys}')
        start = datetime.now()

        r_search.fit(X_data, y_data)
        best_result = r_search.best_params_
        best_score = r_search.best_score_

        end = datetime.now()
        print(f'End - {keys}')
        elapsed_time = end - start

        result = [best_score, elapsed_time, best_result]
        results.append(result)

    print(tabulate(results, headers=['Score', 'Time', 'Result']))

