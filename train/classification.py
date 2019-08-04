import sys
sys.path.append("../")

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegressionCV

from preprocess.split_data import get_split_data, split_data


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
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word',
                                 stop_words='english', token_pattern=r'\b[a-zA-Z]+\b',
                                 ngram_range=(1, top_ngram), max_df=0.9, max_features=3000)
    x = vectorizer.fit_transform(corpus_list)
    return x, vectorizer.get_feature_names()


def transform_xy(df,
                 subject_title_tfidf, title_words, subject_abstract_tfidf, abstract_words):
    

    return X, y

def main():
    df_train, df_test = get_split_data()

    # combine texts in each subjects and find tfidf
    subjects, subject_titles, subject_abstracts = combine_corpus(df_train)
    subject_title_tfidf, title_words = tfidf([subject_titles[sub] for sub in subjects], top_ngram=2)
    subject_abstract_tfidf, abstract_words = tfidf([subject_abstracts[sub] for sub in subjects])

    # train classification
    df_train_lr, df_test_lr = split_data(df_test)
    X_train, y_train = transform_xy(df_train_lr, subject_title_tfidf, title_words,
                                    subject_abstract_tfidf, abstract_words)
    X_test, y_test = transform_xy(df_test_lr, subject_title_tfidf, title_words,
                                  subject_abstract_tfidf, abstract_words)
