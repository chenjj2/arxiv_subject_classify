import numpy as np

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

PATH = '/Users/jingjing/Data/arxiv_subject_classification/'
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = np.load(PATH+'X_train.npy'), np.load(PATH+'y_train.npy'),\
                                   np.load(PATH+'X_test.npy'), np.load(PATH+'y_test.npy')

METHODS = {"SVC": SVC(),
           "Tree":DecisionTreeClassifier(),
           "MLP":MLPClassifier(),
           "LR": LogisticRegression(),
           "SGD": SGDClassifier(),
           "GP": GaussianProcessClassifier()}

def main(X_train=X_TRAIN, y_train=Y_TRAIN, X_test=X_TEST, y_test=Y_TEST):
    print("Train")
    #model = LogisticRegressionCV()
    #model = SGDClassifier()
    for name, model in METHODS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} Accuracy {accuracy_score(y_test, y_pred)}")


if __name__ == '__main__':
    main()