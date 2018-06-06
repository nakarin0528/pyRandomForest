# -*- coding: utf-8 -*-

import numpy as np

from TreeNode import TreeNode as Node
from TreeAnalysis import TreeAnalysis

class DecisionTree(object):

    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        super(DecisionTree, self).__init__()

        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree_analysis = TreeAnalysis()

    # 学習をして決定木を作る
    def fit(self, data, target):
        self.root = Node(self.criterion, self.max_depth, self.random_state)
        self.root.build(data, target, 0, np.unique(target))
        self.feature_importances = self.tree_analysis.get_feature_importances(self.root, data.shape[1])


    # 分類クラスの予測を行う
    def predict(self, data):
        # data: テストデータ
        ans = []
        for d in data:
            ans.append(self.root.predict(d))
        return np.array(ans)

    # 予測結果を返す
    def score(self, data, target):
        return sum(self.predict(data) == target) / float(len(target))

    # 分類木の情報を表示
    def print_tree(self):
        self.root.print_tree(0, " ")
