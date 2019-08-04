from preprocess.split_data import get_split_data
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score


def fit_transformer(df_train, ngram_range=(1, 2), max_features=2000):
    def fit_vectorizer(corpus):
        vectorizer = CountVectorizer(ngram_range=ngram_range,
                                     max_features=max_features,
                                     stop_words='english')
        feature = vectorizer.fit_transform(corpus)
        return feature, vectorizer

    X_title, vectorizer_title = fit_vectorizer(df_train.title)
    X_abstract, vectorizer_abstract = fit_vectorizer(df_train.abstract)
    X = hstack([X_title, X_abstract])

    def transformer(df):
        X_title = vectorizer_title.transform(df.title)
        X_abstract = vectorizer_abstract.transform(df.abstract)
        X = hstack([X_title, X_abstract])
        return X

    return X, transformer


class MultinomNaiveBayes:

    def __init__(self, ngram_range=(1, 2), max_features=2000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.transformer = None
        self.clf = None

    def fit(self, df):
        X, self.transformer = fit_transformer(df, self.ngram_range, self.max_features)
        y = df.subject
        self.clf = MultinomialNB()
        self.clf.fit(X, y)
        return self

    def predict(self, df):
        X = self.transformer(df)
        y_pred = self.clf.predict(X)
        return y_pred


if __name__ == '__main__':
    df_train, df_test = get_split_data('../data/train.json')

    for max_features in [500, 1000, 2000, 5000]:
        model = MultinomNaiveBayes(max_features=max_features)
        model.fit(df_train)
        y_pred = model.predict(df_test)

        y_test = df_test.subject
        score = accuracy_score(y_test, y_pred)
        print(max_features, score)
