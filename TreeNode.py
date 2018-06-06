# -*- coding: utf-8 -*-

import numpy as np

class TreeNode(object):

    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        super(TreeNode, self).__init__()
        self.criterion = criterion          # 分類方法
        self.max_depth = max_depth          # 深さの最大
        self.random_state = random_state    # 乱数のシード設定
        self.depth = None                   # 深さ
        self.left = None                    # 左の子ノード（しきい値未満）
        self.right = None                   # 右の子ノード（しきい値以上）
        self.feature = None                 # 分割する特徴番号
        self.threshold = None               # 分割する閾値
        self.label = None                   # 割り当て立てたクラス番号
        self.impurity = None                # 不純度
        self.gini_index = None              # 分割指数
        self.num_data = None                # 割り当てられたデータ数
        self.num_classes  = None            # クラスの数


    # 木の構築
    def build(self, data, target, depth, ini_num_classes):
        # data: ノードに与えられたデータ
        # target: データの分類クラス
        # depth: 深さ
        # ini_num_classes: 初期のクラス数

        self.depth = depth
        self.num_data = len(target)
        self.num_class = [len(target[target==i]) for i in ini_num_classes]

        # 全データが同一になったら終了
        if len(np.unique(target)) == 1:
            self.label = target[0]
            self.impurity = self.criterion_func(target)
            return

        # 自分のクラスを設定（多数決）
        class_count = {i: len(target[target==i]) for i in np.unique(target)}
        self.label = max(class_count.items(), key=lambda x:x[1])[0]
        self.impurity = self.criterion_func(target)

        num_features = data.shape[1]
        self.gini_index = 0.0

        if self.random_state != None:
            np.random.seed(self.random_state)
        f_loop_order = np.random.permutation(num_features).tolist()
        for f in f_loop_order:
            # 分割候補計算
            data_f = np.unique(data[:, f])              # f番目の特徴量
            points = (data_f[:-1] + data_f[1:]) / 2.0   # 中間値

            # 各分割探索
            for threshold in points:
                # 閾値で２グループに分割
                target_L = target[data[:, f] <= threshold]
                target_R = target[data[:, f] > threshold]

                # 分割後の不純度からGini係数を計算
                value = self.calc_gini_index(target, target_L, target_R)

                # よい分割を記憶
                if self.gini_index < value:
                    self.gini_index = value
                    self.feature = f
                    self.threshold = threshold

        # 更新されない or 一定の深さに到達したら終了
        isEnd = self.gini_index == 0 or depth == self.max_depth
        if isEnd:
            return

        # 左右の子を作成し再帰的に分割
        data_L = data[data[:, self.feature] <= self.threshold]
        target_L = target[data[:, self.feature] <= self.threshold]
        self.left = TreeNode(self.criterion, self.max_depth)
        self.left.build(data_L, target_L, depth+1, ini_num_classes)

        data_R = data[data[:, self.feature] > self.threshold]
        target_R = target[data[:, self.feature] > self.threshold]
        self.right = TreeNode(self.criterion, self.max_depth)
        self.right.build(data_R, target_R, depth+1, ini_num_classes)

    # 分類方法の指定
    def criterion_func(self, target):
        classes = np.unique(target)
        data_count = len(target)

        if self.criterion == "gini":
            value = 1
            for c in classes:
                value -= float(len(target[target == c]) / data_count) ** 2.0
        elif self.criterion == "entropy":
            value = 0
            for c in classes:
                p = float(len(target[target == c])) / data_count
                if p != 0.0:
                    value -= p * np.log2(p)

        return value

    # ジニ係数の計算
    def calc_gini_index(self, target_p, target_cl, target_cr):
        # target_p: データ分類クラスの全体
        # target_cl: データ分類クラスの閾値の左側
        # target_cr: データ分類クラスの閾値の右側

        cri_p = self.criterion_func(target_p)
        cri_cl = self.criterion_func(target_cl)
        cri_cr = self.criterion_func(target_cr)
        return cri_p - len(target_cl)/float(len(target_p))*cri_cl - len(target_cr)/float(len(target_p))*cri_cr

    # # 木の剪定
    # def prune(self, criterion, num_node):
    #     # criterion: 剪定条件
    #     # num_node: 全ノード数
    #
    #     # 自身が葉っぱだったら終了
    #     if self.feature == None:
    #         return
    #
    #     # 子ノードの剪定
    #     self.left.prune(criterion, num_node)
    #     self.right.prune(criterion, num_node)
    #
    #     # 左右が葉っぱだったら剪定
    #     if self.left.feature == None and self.right.feature == None:
    #         # 分割貢献度：GiniIndex * (データ数の割合)
    #         result = self.gini_index * float(self.num_data) / num_node
    #
    #         # 貢献度が条件に対して不十分であれば剪定
    #         if result < criterion:
    #             self.feature = None
    #             self.left = None
    #             self.right = None


    # 入力データの分類先クラスを返す
    def predict(self, data):
        # 自身がノードもしくは深さmaxであるかどうか
        isBottom = self.feature == None or self.depth == self.max_depth
        if isBottom:
            return self.label
        else:
            if data[self.feature] < self.threshold:
                return self.left.predict(data)
            else:
                return self.right.predict(data)


    # 分類条件の出力
    def print_tree(self, depth, TF):
        head = "   " * depth + TF + " -> "

        # 自身がノードもしくは深さmaxであるかどうか
        isBottom = self.feature == None or self.depth == self.max_depth
        if isBottom:
            print(head + "{" + str(self.label) + ": " + str(self.num_data) + "}")
        else:
            print(head + str(self.feature) + " < " + str(self.threshold) + "?")
            self.left.print_tree(depth + 1, "T")
            self.right.print_tree(depth + 1, "F")
