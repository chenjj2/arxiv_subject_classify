import sys
sys.path.append("../")

import spacy
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from preprocess.split_data import get_split_data, split_data
from linear_models import METHODS

NLP = spacy.load("en_core_web_sm")
SPACY_VECTOR_DIM = 96

LE = LabelEncoder()

PATH = '/Users/jingjing/Data/arxiv_subject_classification/word_vector_filtered_sm/'


def clean_text(corpus):
    # clean text
    # remove stop words, and specify legal word pattern
    vectorizer = CountVectorizer(
        strip_accents='unicode', analyzer='word', lowercase=True,
        stop_words='english', token_pattern=r'\b[a-zA-Z]{3,}\b',
        ngram_range=(1, 1), max_df=0.9, max_features=3000)
    vectorizer.fit(corpus)

    return vectorizer


def transform_x(df, title_vectorizer, abstract_vectorizer, title_words, abstract_words):
    # Most Time Consuming Part
    X = []
    to_print = 3
    for i, row in df.iterrows():
        # count word in sentence
        tvec = np.asarray(title_vectorizer.transform([row['title']]).todense()).flatten()
        avec = np.asarray(abstract_vectorizer.transform([row['abstract']]).todense()).flatten()

        # reform sentence with count*word
        #title_reform = ''.join([(word+' ') * count for count, word in zip(tvec, title_words)])
        #abstract_reform = ''.join([(word+' ') * count for count, word in zip(avec, abstract_words)])

        # reform sentence if word exists
        title_reform = ' '.join([word for count, word in zip(tvec, title_words) if count])
        abstract_reform = ' '.join([word for count, word in zip(avec, abstract_words) if count])

        if not title_reform:
            title_reform = ' '
        if not abstract_reform:
            abstract_reform = ' '

        # check reformed sentences
        if to_print:
            print("reformed title/abstract")
            print(title_reform)
            print(abstract_reform)
            to_print -= 1
        # spacy word2vec encode sentence
        xi = np.hstack([NLP(title_reform).vector, NLP(abstract_reform).vector])
        X.append(xi)

    X = np.stack(X, axis=0)
    return X.reshape(len(df), SPACY_VECTOR_DIM*2)


def get_vector(path=PATH):
    print("Read Data")
    df_train, df_test = get_split_data(test_size=0.1)
    print("Use Sub Sample")
    df_train, df_test = split_data(df_test, test_size=0.2)

    # clean text
    print("Clean Text")
    title_vectorizer = clean_text(df_train['title'])
    title_words = title_vectorizer.get_feature_names()
    print("title words", title_words[:10])
    abstract_vectorizer = clean_text(df_train['abstract'])
    abstract_words = abstract_vectorizer.get_feature_names()
    print("abstract words", abstract_words[:10])

    # NOTE: spacy vector simple adds each words vector, so word order is ignored
    print("Transform X Y")
    X_train = transform_x(df_train, title_vectorizer, abstract_vectorizer, title_words, abstract_words)
    X_test = transform_x(df_test, title_vectorizer, abstract_vectorizer, title_words, abstract_words)
    LE.fit_transform(df_train['subject'].append(df_test['subject']))
    y_train = LE.transform(df_train['subject'])
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
    '''
    en_core_web_sm
    SVC: 0.35
    Tree:  0.13
    MLP: 0.29
    LR: 0.35
    SGD: 0.22
    '''
    #get_vector()

    X_train, y_train, X_test, y_test = np.load(PATH + 'X_train.npy'), np.load(PATH + 'y_train.npy'), \
                                       np.load(PATH + 'X_test.npy'), np.load(PATH + 'y_test.npy')

    main(X_train, y_train, X_test, y_test)



