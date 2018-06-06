# -*- coding: utf-8 -*-

import numpy as np

from DecisionTree import DecisionTree

class RandomForest(object):

    def __init__(self, n_estimators=100, random_state=None):
        # n_estimators: 推定数
        # random_state: 乱数のシード

        self.n_estimators = n_estimators
        self.random_state = random_state

    # データからサンプリング
    def reselect_data(self, data, targets):
        # data: 与えられたデータ
        # targets: 分類クラス

        num_data, num_features = data.shape
        if self.random_state != None:
            np.random.seed(self.random_state)

        rs_data_index = np.random.randint(0, num_data, size=num_data)
        rs_num_features = int(np.ceil(np.sqrt(num_features)))
        rs_feature_index = np.random.permutation(num_features)[0:rs_num_features]
        remove_feature_index = [ i for i in set(range(num_features)) - set(rs_feature_index.tolist())]
        rs_data = data[rs_data_index, :]
        rs_data[:, remove_feature_index] = 0.0
        rs_targets = targets[rs_data_index]
        return rs_data, rs_targets

    # 学習して決定木量産
    def fit(self, X_train, y_train):
        num_data, num_features = X_train.shape

        self.estimators = []
        for i in range(self.n_estimators):
            self.estimators.append(DecisionTree(random_state=self.random_state))
            rs_X_train, rs_y_target = self.reselect_data(X_train, y_train)
            self.estimators[i].fit(rs_X_train, rs_y_target)

        self.calc_feature_importances()

    # 特徴量の計算
    def calc_feature_importances(self):
        self.feature_importances = [0.0] * len(self.estimators[0].feature_importances)
        for i in range(self.n_estimators):
            self.feature_importances += self.estimators[i].feature_importances / self.n_estimators

    # 各々の予測
    def predict(self, X_test):
        ans = []
        for i in range(self.n_estimators):
            ans.append(self.estimators[i].predict(X_test).tolist())
        ans = np.array(ans)

        label = []
        for j in range(X_test.shape[0]):
            target = ans[:, j]
            class_count = {i: len(target[target == i]) for i in np.unique(target)}
            label.append(max(class_count.items(), key=lambda x: x[1])[0])

        return np.array(label)

    # 全体の予測
    def score(self, data, target):
        # data: 与えられたデータ
        # targets: 分類クラス

        return np.sum(self.predict(data) == target) / float(len(target))
