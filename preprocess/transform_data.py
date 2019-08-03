from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def encode_subject(train_data):
    subjects = [d['subject'] for d in train_data]
    le = LabelEncoder()
    le.fit(subjects)
    encoded_subjects = le.transform(subjects)
    return encoded_subjects


def encode_date(train_data, resolution='day'):
    if resolution == 'day':
        dates = [d['date'][:10] for d in train_data]
    elif resolution == 'month':
        dates = [d['date'][:7] for d in train_data]
    elif resolution == 'year':
        dates = [d['date'][:4] for d in train_data]
    else:
        raise Exception("resolution should be day/month/year")

    sorted_dates = sorted(dates)
    le = LabelEncoder()
    le.fit(sorted_dates)
    encoded_dates = le.transform(dates)
    return encoded_dates


def combine_corpus(train_data):
    """ return title/abstract corpus for each subject.
    corpus = {"subject": concatenated string}
    """
    title_corpus = defaultdict(str)
    abstract_corpus = defaultdict(str)
    for d in train_data:
        subject = d['subject']
        title_corpus[subject] += d['title']+'\n'
        abstract_corpus[subject] += d['abstract']+'\n'

    return title_corpus, abstract_corpus


def tfidf(corpus):
    """ input list of corpus wrt to each subject, return tfidf and feature_names """
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word',
                                 stop_words='english', token_pattern=r'[a-zA-Z]', ngram_range=(1, 3),
                                 max_df=0.9, max_features=10000)
    x = vectorizer.fit_transform(corpus)
    return x, vectorizer.get_feature_names()