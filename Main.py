import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from RandomForest import RandomForest
from PlotResult import PlotResult

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
    plt = PlotResult(rf, X_train, y_train, X_test, y_test, feature_names, "result")
    plt.plot_result()


if __name__ == "__main__":
    main()
