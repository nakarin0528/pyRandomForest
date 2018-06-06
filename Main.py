import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

from RandomForest import RandomForest

def main():
    use_feature_index = [2 ,3]
    iris = datasets.load_iris()
    X = iris.data[:, use_feature_index]
    y = iris.target
    feature_names = np.array(iris.feature_names)[use_feature_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    n_estimators = 50

    rf = RandomForest(n_estimators=n_estimators, random_state=300)
    rf.fit(X_train, y_train)
    score_m = rf.score(X_test, y_test)

    # scoreの出力
    print("-"*50)
    print("score:" + str(score_m))

    # 特徴量の重要度の出力
    print("-"*50)
    f_importance_m = rf.feature_importances

    print ("feature importances:")
    for f_name, f_importance in zip(feature_names, f_importance_m):
        print("    ",f_name,":", f_importance)

    # 決定した領域の出力
    plot_result(rf, X_train, y_train, X_test, y_test, feature_names, "result ")

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
    # plt.show()
    plt.savefig("win_random_forest_" + png_name + ".png", dpi=300)

if __name__ == "__main__":
    main()
