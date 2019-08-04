import numpy as np

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

PATH = '/Users/jingjing/Data/arxiv_subject_classification/tfidf_sub_sample/'
METHODS = {"SVC": SVC(),
           "Tree":DecisionTreeClassifier(),
           "MLP":MLPClassifier(max_iter=500),
           "LR": LogisticRegression(),
           "SGD": SGDClassifier()}


def main(X_train, y_train, X_test, y_test):

    for name, model in METHODS.items():
        print(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Accuracy {accuracy_score(y_test, y_pred)}")

    model = OneVsRestClassifier(METHODS['MLP'], n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("OVR")
    print(f"Accuracy {accuracy_score(y_test, y_pred)}")

if __name__ == '__main__':
    """
    SCV Accuracy 0.47052154195011336
    Tree Accuracy 0.3956916099773243
    MLP Accuracy 0.572562358276644
    LR Accuracy 0.5527210884353742
    SGD Accuracy 0.5340136054421769
    OVR-MLP(max_iter=500) Accuracy 0.5544217687074829
    """
    X_train, y_train, X_test, y_test = np.load(PATH + 'X_train.npy'), np.load(PATH + 'y_train.npy'), \
                                       np.load(PATH + 'X_test.npy'), np.load(PATH + 'y_test.npy')
    main(X_train, y_train, X_test, y_test)