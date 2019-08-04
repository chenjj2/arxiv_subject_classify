import numpy as np

from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.metrics import accuracy_score

PATH = '/Users/jingjing/Data/arxiv_subject_classification/'
X_train, y_train, X_test, y_test = np.load(PATH+'X_train.npy'), np.load(PATH+'y_train.npy'),\
                                   np.load(PATH+'X_test.npy'), np.load(PATH+'y_test.npy')

def main(X_train, y_train, X_test, y_test):
    print("Train")
    #model = LogisticRegressionCV()
    model = SGDClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy {accuracy_score(y_test, y_pred)}")


if __name__ == '__main__':
    main()