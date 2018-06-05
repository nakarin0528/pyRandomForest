import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn import __version__ as sklearn_version
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if sklearn_version < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

def main():
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
    #--- sklearn random forest
    clf_s = RandomForestClassifier(n_estimators=n_estimators, random_state=500)
    clf_s.fit(X_train, y_train)
    score_s = clf_s.score(X_test, y_test)

    #--- print score
    print("-"*50)
    print("my random forest score:" + str(score_m))
    print("sklearn random forest score:" + str(score_s))

    #---print feature importances
    print("-"*50)
    f_importance_m = clf_m.feature_importances_
    f_importance_s = clf_s.feature_importances_

    print ("my random forest feature importances:")
    for f_name, f_importance in zip(feature_names, f_importance_m):
        print("    ",f_name,":", f_importance)

    print ("sklearn random forest feature importances:")
    for f_name, f_importance in zip(feature_names, f_importance_s):
        print("    ",f_name,":", f_importance)

    #--- output decision region
    plot_result(clf_m, X_train, y_train, X_test, y_test, feature_names, "my_random_forest ")
    plot_result(clf_s, X_train, y_train, X_test, y_test, feature_names, "sklearn_random_forest ")

    
def plot_result(clf, X_train, y_train, X_test, y_test, feature_names, png_name):
    X = np.r_[X_train, X_test]
    y = np.r_[y_train, y_test]
    
    markers = ('s','d', 'x','o', '^', 'v')
    colors = ('green', 'yellow','red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    labels = ('setosa', 'versicolor', 'virginica')

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    dx = 0.02
    X1 = np.arange(x1_min, x1_max, dx)
    X2 = np.arange(x2_min, x2_max, dx)
    X1, X2 = np.meshgrid(X1, X2)
    Z = clf.predict(np.array([X1.ravel(), X2.ravel()]).T)
    Z = Z.reshape(X1.shape)
    
    fig_size_factor = 1.0
    plt.figure(figsize=(12*fig_size_factor, 10*fig_size_factor))
    plt.clf()
    plt.contourf(X1, X2, Z, alpha=0.5, cmap=cmap)
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for idx, cl in enumerate(np.unique(y_train)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=1.0, c=cmap(idx),
                    marker=markers[idx], label=labels[idx])
        
    plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c="", marker="o", s=100,  label="test set")

    plt.title("Decision region(" + png_name + ")")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(loc="upper left")
    plt.grid()
    #plt.show()
    plt.savefig("decision_region_" + png_name + ".png", dpi=300)
    
    
class RandomForest(object):
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        
    def reselect_samples(self, samples, targets):
        num_samples, num_features = samples.shape
        if self.random_state != None:
            np.random.seed(self.random_state)
        rs_sample_index  = np.random.randint(0, num_samples, size=num_samples)
        rs_num_features  = int(np.ceil(np.sqrt(num_features)))
        rs_feature_index = np.random.permutation(num_features)[0:rs_num_features]
        remove_feature_index = [i for i in set(range(num_features)) - set(rs_feature_index.tolist())]
        rs_samples = samples[rs_sample_index, :]
        rs_samples[:, remove_feature_index] = 0.0
        rs_targets = targets[rs_sample_index]
        return rs_samples, rs_targets
    
        
    def fit(self, X_train, y_train):
        num_samples, num_features = X_train.shape

        self.estimators_ = []
        for i in range(self.n_estimators):
            self.estimators_.append(DecisionTree(random_state=self.random_state))
            rs_X_train, rs_y_target = self.reselect_samples(X_train, y_train)
            self.estimators_[i].fit(rs_X_train, rs_y_target)
        
        self.calc_feature_importances()
        
        
    def calc_feature_importances(self):
        self.feature_importances_ = [0.0]*len(self.estimators_[0].feature_importances_)
        for i in range(self.n_estimators):
            self.feature_importances_ += self.estimators_[i].feature_importances_ / self.n_estimators


    def predict(self, X_test):
        pred = []
        for i in range(self.n_estimators):

            pred.append(self.estimators_[i].predict(X_test).tolist())
        pred = np.array(pred)


        label = []
        for j in range(X_test.shape[0]):
            target = pred[:, j]
            class_count = {i: len(target[target == i]) for i in np.unique(target)}
            label.append(max(class_count.items(), key=lambda x: x[1])[0])
        return np.array(label)

    def score(self, sample, target):
        return np.sum(self.predict(sample) == target)/float(len(target))
    
    
class Node(object):
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        self.criterion    = criterion
        self.max_depth    = max_depth
        self.random_state = random_state
        self.depth        = None
        self.left         = None
        self.right        = None
        self.feature      = None
        self.threshold    = None
        self.label        = None
        self.impurity     = None
        self.info_gain    = None
        self.num_samples  = None
        self.num_classes  = None
        
    def split_node(self, sample, target, depth, ini_num_classes):
        self.depth = depth

        self.num_samples = len(target)
        self.num_classes = [len(target[target==i]) for i in ini_num_classes]

        if len(np.unique(target)) == 1:
            self.label = target[0]
            self.impurity = self.criterion_func(target)
            return
        class_count = {i: len(target[target==i]) for i in np.unique(target)}
        self.label = max(class_count.items(), key=lambda x:x[1])[0]
        self.impurity = self.criterion_func(target)
        
        num_features = sample.shape[1]
        self.info_gain = 0.0

        if self.random_state!=None:
            np.random.seed(self.random_state)
        f_loop_order = np.random.permutation(num_features).tolist()
        for f in f_loop_order:
            uniq_feature = np.unique(sample[:, f])
            split_points = (uniq_feature[:-1] + uniq_feature[1:]) / 2.0

            for threshold in split_points:
                target_l = target[sample[:, f] <= threshold] 
                target_r = target[sample[:, f] >  threshold]
                val = self.calc_info_gain(target, target_l, target_r)
                if self.info_gain < val:
                    self.info_gain = val
                    self.feature   = f
                    self.threshold = threshold

        if self.info_gain == 0.0:
            return
        if depth == self.max_depth:
            return

        sample_l   = sample[sample[:, self.feature] <= self.threshold]
        target_l   = target[sample[:, self.feature] <= self.threshold]
        self.left  = Node(self.criterion, self.max_depth)
        self.left.split_node(sample_l, target_l, depth + 1, ini_num_classes)

        sample_r   = sample[sample[:, self.feature] > self.threshold]
        target_r   = target[sample[:, self.feature] > self.threshold]
        self.right = Node(self.criterion, self.max_depth)
        self.right.split_node(sample_r, target_r, depth + 1, ini_num_classes)

    def criterion_func(self, target):
        classes = np.unique(target)
        numdata = len(target)

        if self.criterion == "gini":
            val = 1
            for c in classes:
                p = float(len(target[target == c])) / numdata
                val -= p ** 2.0
        elif self.criterion == "entropy":
            val = 0
            for c in classes:
                p = float(len(target[target == c])) / numdata
                if p!=0.0:
                    val -= p * np.log2(p)
        return val

    def calc_info_gain(self, target_p, target_cl, target_cr):
        cri_p  = self.criterion_func(target_p)
        cri_cl = self.criterion_func(target_cl)
        cri_cr = self.criterion_func(target_cr)
        return cri_p - len(target_cl)/float(len(target_p))*cri_cl - len(target_cr)/float(len(target_p))*cri_cr

    def predict(self, sample):
        if self.feature == None or self.depth == self.max_depth:
            return self.label
        else:
            if sample[self.feature] <= self.threshold:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)

    
class TreeAnalysis(object):
    def __init__(self):
        self.num_features = None
        self.importances  = None
        
    def compute_feature_importances(self, node):
        if node.feature == None:
            return
        
        self.importances[node.feature] += node.info_gain*node.num_samples
        
        self.compute_feature_importances(node.left)
        self.compute_feature_importances(node.right)
        
    def get_feature_importances(self, node, num_features, normalize=True):
        self.num_features = num_features
        self.importances  = np.zeros(num_features)
        
        self.compute_feature_importances(node)
        self.importances /= node.num_samples

        if normalize:
            normalizer = np.sum(self.importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                self.importances /= normalizer
        return self.importances
    
            
class DecisionTree(object):
    def __init__(self, criterion="gini", max_depth=None, random_state=None):
        self.tree          = None
        self.criterion     = criterion
        self.max_depth     = max_depth
        self.random_state  = random_state
        self.tree_analysis = TreeAnalysis()

    def fit(self, sample, target):
        self.tree = Node(self.criterion, self.max_depth, self.random_state)
        self.tree.split_node(sample, target, 0, np.unique(target))
        self.feature_importances_ = self.tree_analysis.get_feature_importances(self.tree, sample.shape[1])

    def predict(self, sample):
        pred = []
        for s in sample:
            pred.append(self.tree.predict(s))
        return np.array(pred)

    def score(self, sample, target):
        return sum(self.predict(sample) == target)/float(len(target))

        
if __name__ == "__main__":
    main()