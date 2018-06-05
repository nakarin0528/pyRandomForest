# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

class RandomForest:

    def __init__(self, n_estimators=100, random_state=None) :
        self.n_estimators = n_estimators
        self.random_state = random_state

    def reselect_samples(self, samples, targets):
        num_samples, num_features = samples.shape
        if self.random_state != None:
            np.random.seed(self.random_state)
        rs_sample_index = np.random.ranint(0, num_samples, size = num_samples)
        rs_num_features = int(np.ceil(np.sqrt(num_features)))
        rs_feature_index = np.random.permutation(num_features)[0:rs_num_features]
        remove_feature_index = [ i for i in set(range(num_features)) - set(rs_feature_index.tolist())]
        rs_samples = samples[rs_sample_index, :]
        rs_samples = [:, remove_feature_index] = 0.0
        rs_targets = targets[rs_sample_index]
        return rs_samples, rs_targets

    def fit(self, X_train, y_train):
        num_samples, num_features = X_train.shape

        self.estimators = []
        for i in range(self.n_estimators):
            self.estimators.append(DecisionTree())


if __name__ == "__main__":
    use_feature_index = [2 ,3]
    iris = datasets.load_iris()
    X = iris.data[:, use_feature_index]
    y = iris.target
    feature_names = np.array(iris.feature_names)[use_feature_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    n_estimators = 50

    #--- my random forest
    clf_m = RandomForest(n_estimators=n_estimators, random_state=300)
    clf_m.fit(X_train, y_train)
    score_m = clf_m.score(X_test, y_test)

    #--- print score
    print("-"*50)
    print("my random forest score:" + str(score_m))

    #---print feature importances
    print("-"*50)
    f_importance_m = clf_m.feature_importances_

    print ("my random forest feature importances:")
    for f_name, f_importance in zip(feature_names, f_importance_m):
        print("    ",f_name,":", f_importance)

    #--- output decision region
    plot_result(clf_m, X_train, y_train, X_test, y_test, feature_names, "my_random_forest ")
