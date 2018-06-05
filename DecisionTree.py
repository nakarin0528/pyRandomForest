# -*- coding: utf-8 -*-

import numpy as np
import TreeNode as Node

class DecisionTree:

    def __init__(self, criterion=0.1):
        super(DecisionTree, self).__init__()

        self.root = None
        self.criterion = criterion

    # 学をして決定木を作る
    def fit(self, data, target):
        self.root = Node()
        self.root.build(data, target)
        self.root.prune(self.criterion, self.root.num_data)


    # 分類クラスの予測を行う
    def predict(self, data) :
        # data: テストデータ
        ans = []
        for d in data:
            ans.append(self.root.predict(d))
        return np.array(ans)

    # 分類木の情報を表示
    def print_tree(self) :
        self.root.print_tree(0, " ")
