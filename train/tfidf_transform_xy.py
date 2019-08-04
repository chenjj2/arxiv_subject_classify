import sys
sys.path.append("../")

import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


from preprocess.split_data import get_split_data, split_data

PATH = '/Users/jingjing/Data/arxiv_subject_classification/'


def combine_corpus(df_train):
    unique_subjects = df_train['subject'].unique().tolist()

    title_corpus = defaultdict(str)
    abstract_corpus = defaultdict(str)
    for sub in unique_subjects:
        titles = df_train[df_train['subject']==sub]['title']
        title_corpus[sub] = '\n'.join(titles.tolist())
        abstracts = df_train[df_train['subject'] == sub]['abstract']
        abstract_corpus[sub] = '\n'.join(abstracts.tolist())

    return unique_subjects, title_corpus, abstract_corpus


def tfidf(corpus_list, top_ngram=1):
    vectorizer = TfidfVectorizer(
        strip_accents='unicode', analyzer='word',
        stop_words='english', token_pattern=r'\b[a-zA-Z]+\b',
        ngram_range=(1, top_ngram), max_df=0.9, max_features=3000)
    x = vectorizer.fit_transform(corpus_list)
    return x, vectorizer


def transform_xy(df, title_matrix, title_vectorizer,
                 abstract_matrix, abstract_vectorizer, subjects):

    df_len = len(df)
    sub_len = len(subjects)

    X, y = [], []
    for i, row in df.iterrows():
        tvec, avec = title_vectorizer.transform([row['title']]), \
                     abstract_vectorizer.transform([row['abstract']])

        Xi = np.hstack([np.dot(title_matrix, tvec.T).T.toarray(),
                        np.dot(abstract_matrix, avec.T).T.toarray()])
        yi = subjects.index(row['subject'])
        X.append(Xi); y.append(yi)
    return np.array(X).reshape(df_len, sub_len*2), np.array(y)


def save_xy(path=PATH):
    print("Read Data")
    df_train, df_test = get_split_data()

    # combine texts in each subjects and find tfidf
    print("Combine Corpus")
    subjects, subject_titles, subject_abstracts = combine_corpus(df_train)
    title_matrix, title_vectorizer = tfidf([subject_titles[sub] for sub in subjects], top_ngram=2)
    abstract_matrix, abstract_vectorizer = tfidf([subject_abstracts[sub] for sub in subjects])

    # train classification
    print("Transform XY")
    #df_train_lr, df_test_lr = split_data(df_test)
    df_train_lr, df_test_lr = df_train, df_test
    X_train, y_train = transform_xy(df_train_lr, title_matrix, title_vectorizer,
                                    abstract_matrix, abstract_vectorizer, subjects)
    X_test, y_test = transform_xy(df_test_lr, title_matrix, title_vectorizer,
                                  abstract_matrix, abstract_vectorizer, subjects)

    print("Save to File")
    np.save(path+'X_train.npy', X_train)
    np.save(path + 'y_train.npy', y_train)
    np.save(path + 'X_test.npy', X_test)
    np.save(path + 'y_test.npy', y_test)


if __name__ == '__main__':
    save_xy()
