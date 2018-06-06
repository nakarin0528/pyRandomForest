import numpy as np

from sklearn import datasets
from sklearn import __version__ as sklearn_version
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

if sklearn_version < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

from PlotResult import PlotResult

def main():
    use_feature_index = [2 ,3]
    iris = datasets.load_iris()
    X = iris.data[:, use_feature_index]
    y = iris.target
    feature_names = np.array(iris.feature_names)[use_feature_index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    n_estimators = 50

    clf_s = RandomForestClassifier(n_estimators=n_estimators, random_state=500)
    clf_s.fit(X_train, y_train)
    score_s = clf_s.score(X_test, y_test)

    # scoreの出力
    print("-"*50)
    print("sklearn random forest score:" + str(score_s))

    # 特徴量の重要度の出力
    print("-"*50)
    f_importance_s = clf_s.feature_importances_

    print ("sklearn random forest feature importances:")
    for f_name, f_importance in zip(feature_names, f_importance_s):
        print("    ",f_name,":", f_importance)

    # 決定した領域の出力
    plt = PlotResult(clf_s, X_train, y_train, X_test, y_test, feature_names, "sklearn_random_forest")
    plt.plot_result()


if __name__ == "__main__":
    main()
