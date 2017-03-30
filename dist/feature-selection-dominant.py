# Author: Marius Maaland
#         Jonas Palm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

import dataset

preprocess=True #("emails", "headers")
y_train, X_train_data = dataset.get(subset="train", preprocess=preprocess, verbose=True)
y_test, X_test_data = dataset.get(subset="test", preprocess=preprocess)

VEC_MAX_DF=1.0
VEC_MIN_DF=1
VEC_STOP_WORDS='english'

def print_dominant_words(vec, n):
    X_train = vec.fit_transform(X_train_data)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    labels = np.unique(y_train)
    coefs = clf.coef_.argsort()[:, ::-1]
    terms = vec.get_feature_names()
    for i in range(len(labels)):
        print "{:>28}:".format(labels[i]),
        for ind in coefs[i, :10]:
            print " (%s)" % terms[ind],
        print ""
        


vec1 = TfidfVectorizer(max_df=VEC_MAX_DF, max_features=10000,
                           min_df=VEC_MIN_DF, stop_words=VEC_STOP_WORDS,
                           use_idf=True, ngram_range=(1,1))

vec12 = TfidfVectorizer(max_df=VEC_MAX_DF, max_features=10000,
                            min_df=VEC_MIN_DF, stop_words=VEC_STOP_WORDS,
                            use_idf=True, ngram_range=(1,2))

vec2 = TfidfVectorizer(max_df=VEC_MAX_DF, max_features=10000,
                           min_df=VEC_MIN_DF, stop_words=VEC_STOP_WORDS,
                           use_idf=True, ngram_range=(2,2))

print 80*"-"
print "tfidf 1-gram:"
print_dominant_words(vec1, 10)
print 80*"-"
print "tfidf 1,2-gram:"
print_dominant_words(vec12, 10)
print 80*"-"
print "tfidf 2-gram:"
print_dominant_words(vec2, 10)
