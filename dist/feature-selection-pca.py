from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn import manifold
from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dataset


categories = ['rec.sport.hockey', 'talk.politics.guns', 'comp.graphics', 'sci.crypt', 'sci.electronics']

def plot(preprocess):
    labels, data = dataset.get(subset="train",
                                   preprocess=preprocess,
                                   categories=categories,
                                   verbose=True)
    labels = np.array(labels)

    print "Getting TF IDF weights"
    
    vec = TfidfVectorizer(max_df=0.5, max_features=10000,
                            min_df=2, stop_words='english',
                            use_idf=True, ngram_range=(1,1))
    X = vec.fit_transform(data)

    print(repr(X))

    print "Reducing dimensions to 50"

    X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)

    X_embedded = PCA(n_components=2).fit_transform(X_reduced)


    names = np.unique(labels)
    print names
    num_clusters = len(names)
    fig = plt.figure(frameon=False)

    colors = iter(cm.Spectral(np.linspace(0, 1, num_clusters)))

    for name in names:
        X = X_embedded[labels == name]
        plt.scatter(X[:,0], X[:,1], marker='x', label=name)

    plt.title("PCA (Preprocessed)" if preprocess else "PCA")
    plt.xticks([])
    plt.yticks([])
    plt.legend()

plot(False)
plot(True)
plt.show()

#plt.scatter(Y[:,0], Y[:, 1], cmap=plt.cm.Spectral)

