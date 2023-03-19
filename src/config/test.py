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
from datetime import datetime
import csv

def test():
    with open('config_models.yml') as f:
        my_dict = yaml.safe_load(f)
    
    with open('../data/gender.pkl', 'rb') as f:
        # X_train, X_test, y_train, y_test = pickle.load(f)
        X_train, y_train, X_test, y_test = pickle.load(f)
    X_data = np.concatenate((X_train, X_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    
    results = []
    iters = [2, 5, 10, 20, 50]
    for iter in iters:
        random_search = RandomizedSearchCV(estimator=SVC(tol=0.01), param_distributions=my_dict['svc'], scoring='accuracy', cv=10, n_iter=iter, verbose=1)
        
        loader = Loader('Loading', 'Finished', 0.05).start()
        start = datetime.now()
        random_search.fit(X_data, y_data)
        end = datetime.now()
        elapsed = end-start
        loader.stop()

        best_param = random_search.best_params_
        best_result = random_search.best_score_
        print("Best Param: {} and Best Socore: {}".format(best_param, best_result))
        r = [elapsed, 10, iter, best_result, best_param]
        print(r)
        results.append(r)

    columns = ['time', 'cv', 'n_iter', 'accuracy', 'config']
    with open('results.csv', 'wb') as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(columns)
        filewriter.writerows(results)

if __name__ == "__main__":
    test()
