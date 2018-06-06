# -*- coding: utf-8 -*-

import numpy as np

class TreeAnalysis(object):
    def __init__(self):
        self.num_features = None
        self.importances = None

    # 全ての特徴量の重要度を調べる
    def find_feature_importances(self, node):
        if node.feature == None:
            return

        self.importances[node.feature] += node.gini_index * node.num_data

        # 左右の子を再帰的に探索
        self.find_feature_importances(node.left)
        self.find_feature_importances(node.right)

    # 特徴量の重要度を返す
    def get_feature_importances(self, node, num_features, normalize=True):
        self.num_features = num_features
        self.importances = np.zeros(num_features)

        self.find_feature_importances(node)
        self.importances /= node.num_data

        if normalize:
            normalizer = np.sum(self.importances)

            if normalizer > 0.0:
                self.importances /= normalizer

        return self.importances
