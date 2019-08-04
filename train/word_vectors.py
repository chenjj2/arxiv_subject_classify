import sys
sys.path.append("../")

import spacy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from preprocess.split_data import get_split_data, split_data
from linear_models import METHODS

NLP = spacy.load('en_core_web_sm')
SPACY_VECTOR_DIM = 96
LE = LabelEncoder()
PATH = '/Users/jingjing/Data/arxiv_subject_classification/word_vector_sample/'


def get_vector(path=PATH):
    print("Get Vector")
    df_train, df_test = get_split_data()

    print("Use Sub Sample")
    df_train, df_test = split_data(df_test, test_size=0.2)

    X_train = []
    for i, row in df_train.iterrows():
        xi = np.hstack([NLP(row['title']).vector, NLP(row['abstract']).vector])
        X_train.append(xi)

    X_test = []
    for i, row in df_test.iterrows():
        xi = np.hstack([NLP(row['title']).vector, NLP(row['abstract']).vector])
        X_test.append(xi)

    X_train = np.array(X_train).reshape(len(df_train), SPACY_VECTOR_DIM*2)
    X_test = np.array(X_test).reshape(len(df_test), SPACY_VECTOR_DIM*2)
    y_train = LE.fit_transform(df_train['subject'])
    y_test = LE.transform(df_test['subject'])

    print("Save to File")
    np.save(path + 'X_train.npy', X_train)
    np.save(path + 'y_train.npy', y_train)
    np.save(path + 'X_test.npy', X_test)
    np.save(path + 'y_test.npy', y_test)


def main(X_train, y_train, X_test, y_test):

    print("Train")
    for name, model in METHODS.items():
        print(name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Accuracy {accuracy_score(y_test, y_pred)}")


if __name__ == '__main__':
    #get_vector()
    '''
    SVC: 0.31
    Tree: 0.13
    MLP: 0.24
    LR: 0.34
    SGD: 0.26
    '''
    X_train, y_train, X_test, y_test = np.load(PATH + 'X_train.npy'), np.load(PATH + 'y_train.npy'), \
                                       np.load(PATH + 'X_test.npy'), np.load(PATH + 'y_test.npy')

    main(X_train, y_train, X_test, y_test)