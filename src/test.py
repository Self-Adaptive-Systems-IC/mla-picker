#!/usr/bin/env python
import yaml
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import datetime as dt
import multiprocessing as mp
from multiprocessing import Process, current_process
import sys
import time
import csv
from utils.Loader import Loader
import json

def test():
    with open('./src/config/config_models.yml') as f:
        my_dict = yaml.safe_load(f)
    
    with open('./src/data/data.pkl', 'rb') as f:
        # X_train, X_test, y_train, y_test = pickle.load(f)
        X_train, y_train, X_test, y_test = pickle.load(f)
    X_data = np.concatenate((X_train, X_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    
    results = []
    iters = [2, 5, 10, 20, 50]
    for iter in iters:
        # random_search = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=my_dict['decision_tree'], scoring='accuracy', cv=10, n_iter=iter)
        #random_search = RandomizedSearchCV(estimator=RandomForestClassifiedr(), param_distributions=my_dict['random_forest'], scoring='accuracy', cv=10, n_iter=iter)
        random_search = RandomizedSearchCV(estimator=SVC(cache_size=10240), param_distributions=my_dict['svc'], scoring='accuracy', cv=10, n_iter=iter)
        
        loader = Loader(f'Loading {10*iter} fits', 'Finished', 0.05).start()
        start = time.time()
        random_search.fit(X_data, y_data)
        end = time.time()
        elapsed = end-start
        loader.stop()

        best_param = random_search.best_params_
        best_result = random_search.best_score_
        # print("Best Param: {} and Best Socore: {}".format(best_param, best_result))
        
        
        # print(f'\n{elapsed}')
        
        r = [str(elapsed), str(10), str(iter), str(10*iter), str(best_result), json.dumps(best_param)]
        # r = [str(elapsed), str(10), str(iter), str(best_result), json.dumps(best_param)]
        # print(f"\n{r}\n")
        results.append(r)

    columns = ['time', 'cv', 'n_iter', 'fits', 'accuracy', 'config']
    # with open('results_randomizedcv_decision_tree.csv', 'w') as csv_file:
    # with open('results_randomizedcv_random_forest.csv', 'w') as csv_file:
    with open('results_randomizedcv_svc.csv', 'w') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(columns)
        filewriter.writerows(results)

if __name__ == "__main__":
    test()
