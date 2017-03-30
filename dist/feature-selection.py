# Author: Marius Maaland
#         Jonas Palm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

import dataset

preprocess=("emails", "headers")
y_train, X_train_data = dataset.get(subset="train", preprocess=preprocess, verbose=True)
y_test, X_test_data = dataset.get(subset="test", preprocess=preprocess)

VEC_MAX_DF=1.0
VEC_MIN_DF=1
VEC_STOP_WORDS='english'

def benchmark(vec, verbose=False):
    X_train = vec.fit_transform(X_train_data)

    clf = LogisticRegression(verbose=verbose)
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    train_score = metrics.accuracy_score(y_train, train_pred)

    X_test = vec.transform(X_test_data)
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    return train_score, score

vectorizers = {}
for nfeatures in range(1000, 5000, 1000) + range(5000, 25000, 5000):
    """
    name = ("cv-1g", nfeatures)
    vectorizers[name] = CountVectorizer(max_df=VEC_MAX_DF, min_df=VEC_MIN_DF,
                                            max_features=nfeatures,
                                            stop_words=VEC_STOP_WORDS,
                                            ngram_range=(1,1))

    name = ("cv-2g", nfeatures)
    vectorizers[name] = CountVectorizer(max_df=VEC_MAX_DF, min_df=VEC_MIN_DF,
                                            max_features=nfeatures,
                                            stop_words=VEC_STOP_WORDS,
                                            ngram_range=(2,2))

    name = ("cv-3g", nfeatures)
    vectorizers[name] = CountVectorizer(max_df=VEC_MAX_DF, min_df=VEC_MIN_DF,
                                            max_features=nfeatures,
                                            stop_words=VEC_STOP_WORDS,
                                            ngram_range=(3,3))
    """

    name = ("tf-idf-1g", nfeatures)
    vectorizers[name] = TfidfVectorizer(max_df=VEC_MAX_DF, max_features=nfeatures,
                                    min_df=VEC_MIN_DF, stop_words=VEC_STOP_WORDS,
                                    use_idf=True, ngram_range=(1,1))

    name = ("tf-idf-12g", nfeatures)
    vectorizers[name] = TfidfVectorizer(max_df=VEC_MAX_DF, max_features=nfeatures,
                                    min_df=VEC_MIN_DF, stop_words=VEC_STOP_WORDS,
                                    use_idf=True, ngram_range=(1,2))

    name = ("tf-idf-2g", nfeatures)
    vectorizers[name] = TfidfVectorizer(max_df=VEC_MAX_DF, max_features=nfeatures,
                                    min_df=VEC_MIN_DF, stop_words=VEC_STOP_WORDS,
                                    use_idf=True, ngram_range=(2,2))

    name = ("tf-idf-3g", nfeatures)
    vectorizers[name] = TfidfVectorizer(max_df=VEC_MAX_DF, max_features=nfeatures,
                                    min_df=VEC_MIN_DF, stop_words=VEC_STOP_WORDS,
                                    use_idf=True, ngram_range=(3,3))

    name = ("tf-idf-4g", nfeatures)
    vectorizers[name] = TfidfVectorizer(max_df=VEC_MAX_DF, max_features=nfeatures,
                                    min_df=VEC_MIN_DF, stop_words=VEC_STOP_WORDS,
                                    use_idf=True, ngram_range=(4,4))

print "MAX_DF=", VEC_MAX_DF, ", MIN_DF=", VEC_MIN_DF, ", STOP_WORDS=", VEC_STOP_WORDS
print "vectorizer,num_features,train_accuracy,test_accuracy"
for vec in sorted(vectorizers.keys()):
    train_acc, test_acc = benchmark(vectorizers[vec])
    print ", ".join(map(str, [vec[0], vec[1], train_acc, test_acc]))
