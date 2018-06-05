# -*- coding: utf-8 -*-

import numpy as np

class TreeNode:

    def __init__(self, max_depth=None, random_state=None):
        super(TreeNode, self).__init__()
        self.max_depth = max_depth          # 深さの最大
        self.random_state = random_state    # 乱数のシード設定
        self.depth = None                   # 深さ
        self.left = None                    # 左の子ノード（しきい値未満）
        self.right = None                   # 右の子ノード（しきい値以上）
        self.feature = None                 # 分割する特徴番号
        self.threshold = None               # 分割する閾値
        self.label = None                   # 割り当て立てたクラス番号
        self.num_data = None                # 割り当てられたデータ数
        self.gini_index = None              # 分割指数

    # 木の構築
    def build(self, data, target) :
        # data: ノードに与えられたデータ
        # target: データの分類クラス

        self.num_data = data.shape[0]
        num_features = data.shape[1]

        # 全データが同一になったら終了
        if len(np.unique(target)) == 1:
            self.label = target[0]
            return

        # 自分のクラスを設定（多数決）
        num_class = {i: len(target[target==i]) for i in np.unique(target)}
        self.label = max(num_class.items(), key=lambda x:x[1])[0]

        # 最良の分割を記憶する変数
        best_gini_index = 0.0
        best_feature = None
        best_threshold = None

        # 自身の不純度を先に計算
        gini = self.gini_func(target)

        for f in range (num_features):
            # 分割候補計算
            data_f = np.unique(data[:, f])              # f番目の特徴量
            points = (data_f[:-1] + data_f[1:]) / 2.0   # 中間値

            # 各分割探索
            for threshold in points:
                # 閾値で２グループに分割
                target_L = target[data[:, f] < threshold]
                target_R = target[data[:, f] >= threshold]

                # 分割後の不純度からGini係数を計算
                gini_L = self.gini_func(target_L)
                gini_R = self.gini_func(target_R)
                pl = float(target_L.shape[0]) / self.num_data
                pr = float(target_R.shape[0]) / self.num_data
                gini_index = gini - (pl * gini_L + pr * gini_R)

                # よい分割を記憶
                if gini_index > best_gini_index:
                    best_gini_index = gini_index
                    best_feature = f
                    best_threshold = threshold

        # 不純度が減らなければ終了
        if best_gini_index == 0:
            return

        # 最良の分割を保持
        self.feature = best_feature
        self.gini_index = best_gini_index
        self.threshold = best_threshold

        # 左右の子を作成し再帰的に分割
        data_L = data[data[:, self.feature] < self.threshold]
        target_L = target[data[:, self.feature] < self.threshold]
        self.left = TreeNode()
        self.left.build(data_L, target_L)

        data_R = data[data[:, self.feature] >= self.threshold]
        target_R = target[data[:, self.feature] >= self.threshold]
        self.right = TreeNode()
        self.right.build(data_R, target_R)

    # ジニ係数の計算
    def gini_func(self, target):
        # target: 各データの分類クラス
        classes = np.unique(target)
        num_data = target.shape[0]

        gini = 1.0
        for c in classes:
            gini -= (len(target[target == c]) / num_data) ** 2.0

        return gini

    # 木の剪定
    def prune(self, criterion, num_node) :
        # criterion: 剪定条件
        # num_node: 全ノード数

        # 自身が葉っぱだったら終了
        if self.feature == None:
            return

        # 子ノードの剪定
        self.left.prune(criterion, num_node)
        self.right.prune(criterion, num_node)

        # 左右が葉っぱだったら剪定
        if self.left.feature == None and self.right.feature == None:
            # 分割貢献度：GiniIndex * (データ数の割合)
            result = self.gini_index * float(self.num_data) / num_node

            # 貢献度が条件に対して不十分であれば剪定
            if result < criterion:
                self.feature = None
                self.left = None
                self.right = None

    # 入力データの分類先クラスを返す
    def predict(self, data) :
        # 自身がノードの時は条件判定
        if self.feature != None:
            if data[self.feature] < self.threshold:
                return self.left.predict(data)
            else:
                return self.right.predict(data)

        # 自身が葉っぱの時は地震の分類クラスを返す
        else:
            return self.label

    # 分類条件の出力
    def print_tree(self, depth, TF) :
        head = "   " * depth + TF + " -> "

        # ノードの場合
        if self.feature != None:
            print(head + str(self.feature) + " < " + str(self.threshold) + "?")
            self.left.print_tree(depth + 1, "T")
            self.right.print_tree(depth + 1, "F")
        # 葉っぱの場合
        else:
            print(head + "{" + str(self.label) + ": " + str(self.num_data) + "}")
