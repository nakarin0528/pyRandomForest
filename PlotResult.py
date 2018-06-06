import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class PlotResult(object):
    def __init__(self, clf, X_train, y_train, X_test, y_test, feature_names, png_name):
        self.clf = clf
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.png_name = png_name
        
    def plot_result(self):
        X = np.r_[self.X_train, self.X_test]
        y = np.r_[self.y_train, self.y_test]

        markers = ('s','d', 'x','o', '^', 'v')
        colors = ('green', 'blue','red', 'yellow', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        labels = ('setosa', 'versicolor', 'virginica')

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        dx = 0.02
        X1 = np.arange(x1_min, x1_max, dx)
        X2 = np.arange(x2_min, x2_max, dx)
        X1, X2 = np.meshgrid(X1, X2)
        Z = self.clf.predict(np.array([X1.ravel(), X2.ravel()]).T)
        Z = Z.reshape(X1.shape)

        fig_size_factor = 1.0
        plt.figure(figsize=(12*fig_size_factor, 10*fig_size_factor))
        plt.clf()
        plt.contourf(X1, X2, Z, alpha=0.5, cmap=cmap)
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())

        for idx, cl in enumerate(np.unique(self.y_train)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=1.0, c=cmap(idx),
                        marker=markers[idx], label=labels[idx])

        plt.scatter(x=self.X_test[:, 0], y=self.X_test[:, 1], c="", marker="o", s=100,  label="test set")

        plt.title("Decision region(" + self.png_name + ")")
        plt.xlabel(self.feature_names[0])
        plt.ylabel(self.feature_names[1])
        plt.legend(loc="upper left")
        plt.grid()
        # plt.show()
        plt.savefig("random_forest_" + self.png_name + ".png", dpi=300)