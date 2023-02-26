from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from utils.Loader import Loader

def cross_validation(X_data, y_data):
    """
    Function to execute grid search for decision tree and random forest.
    Return an array of array with the accuracy for each algorithm

    Parameters
    ----------
        X_data : array
            The data to fit. Can be a list or an array.        
        y_data : array
            The target variable to try predict in the case of supervised learning.
    """
    n_size = 10
    results_tree = []
    results_random_forest = []
    results_svm = []
    loader = Loader(
            f"・ Running \033[1m cross validation \033[0m",
            f"・ The \033[1m cross validation \033[0m is done!!!",
            0.05
        ).start()
    for i in range(n_size):
        kfold = KFold(n_splits=10, shuffle=True, random_state=i)
        # For decision tree
        decision_tree = DecisionTreeClassifier(criterion='log_loss', min_samples_leaf=15, min_samples_split=25, splitter='random')
        scores_tree = cross_val_score(decision_tree, X_data, y_data, cv=kfold)
        results_tree.append(scores_tree.mean())
        # For random forest   
        random_forest = RandomForestClassifier(n_estimators=40,  criterion='gini', min_samples_leaf=1, min_samples_split=10)
        scores_random_forest = cross_val_score(random_forest, X_data, y_data, cv=kfold)
        results_random_forest.append(scores_random_forest.mean())
        # For svc
        svm = SVC(kernel = 'rbf', C = 100.0, tol=0.001)
        scores_svm = cross_val_score(svm, X_data, y_data, cv=kfold)
        results_svm.append(scores_svm.mean())
    loader.stop()
    # results = [results_tree, results_random_forest]
    results = [results_tree, results_random_forest, results_svm]
    return results
