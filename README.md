# Classify arXiv Articles
## Problem 
Classify an arXiv Article's Subject based on its Title and Abstract

## Method and Performance

We performed training/test split with ratio 9 : 1.

#### 1. Multinomial Naive Bayes
The workflow is as follows:
1. For training data, fit CountVectorizer using title and abstract, respectively.
2. Concat title features and abstract features to get X_train.
3. Fit (X_train, y_train) using multinomial naive Bayes method, where y_train is subject.
4. For test data, use the vectorizer in Step 1 followed by Step 2 to get X_test.
5. Get prediction y_pred from X_test.

For ngram_range = (1, 2), we tested on the following max_features:

| max features | accuracy score (%) |
| --- | --- |
| 100 | 41.75 |
| 200 | 43.95 |
| 500 | 47.90 |
| 1000 | 49.64 |
| 2000 | 51.63 |
| 5000 | 54.51 |