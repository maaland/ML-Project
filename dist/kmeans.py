import numpy as np
import timeit
import warnings
import sys
import matplotlib.pyplot as plt

class KMeans:
    """
    K-Means implementation.
    
    This class is capable of doing K-Means, K-Means++ and MiniBatch K-Means.

    Initialization arguments:
      n_cluster - Number of clusters (K).
      n_starts - Number of restarts. Controls how many times the
                 algorithm will be run. The result will be the run that
                 performs the best.
      init - "k-means" or "k-means++", controls wether to use k-means++ initialization
             or not.
      minibatch - Controls wether to use K-means minibatch or not.

    Training example (use it like you would use a Scikit classifier):

       km = Kmeans(...)
       km.fit(X)

       print km.labels_
       print km.centroids_

       # other metrics
       print km.avg_iteration # avg iterations per run
       print km.avg_time # avg time per run
       
    """
    def __init__(self, n_clusters = 2, n_starts = 5, init = "k-means", minibatch=False, verbose=False):
        if init == "k-means":
            self.pp = False
        elif init == "k-means++":
            self.pp = True
        else:
            raise Exception("Invalid init argument")

        self.use_minibatch = minibatch
        self.batch_size = 100 # only used for minibatch
        self.max_iterations = 100
        self.verbose = verbose

        if self.verbose:
            print "K-means++?", self.pp
            print "Minibatch?", self.use_minibatch

        # number of restarts
        self.n_starts = n_starts

        # K in K-means :)
        self.n_clusters = n_clusters

    def cluster(self, X):
        C = self.initialize(X)
        N = X.shape[0]
        z = np.repeat(-1, N)
        t = 0
        while True:
            t += 1
            old_z = z

            # find closest center for each data point
            z = np.array([np.argmin([np.dot(x-k, x-k) for k in C]) for x in X])
            C = self.update(X, z)

            if np.all(z == old_z) or t >= self.max_iterations:
                # nothing changed, we've converged
                dist = self.compute_distortion(X, C, z)
                return dict(centroids=C, z=z, dist=dist, iters=t)

    def minibatch(self, X):
        b = self.batch_size
        C = self.initialize(X)

        N, D = X.shape

        # number of iterations
        t = 0
        # initialize per center counts to zero
        v = np.zeros(self.n_clusters)
        while True:
            t += 1
            # pick b examples randomly from X
            M = X[np.random.choice(N, self.batch_size)]
            # cache the center nearest to each x in M 
            d = np.array([np.argmin([np.dot(x-k, x-k) for k in C]) for x in M])
            
            # update center for each data point
            for i in range(len(M)):
                # get center index and data point
                c, x = d[i], M[i]
                # update per center count
                v[c] = v[c] + 1
                # per center learning rate
                nn = 1.0 / v[c]
                # update center
                C[c] = (1 - nn)*C[c] + nn * x

            # we done?
            if t >= self.max_iterations:
                # compute labels for all data points
                z = np.array([np.argmin([np.dot(x-k, x-k) for k in C]) for x in X])
                dist = self.compute_distortion(X, C, z)
                return dict(centroids=C, z=z, dist=dist, iters=t)

    def fit(self, X):
        """
        Run KMeans (using minibatch if self.minibatch is true) n_start
        times and return the run that achieves the lowest within-group
        sum of squares.
        """

        # convert to dense array
        X = X.toarray()

        f = self.cluster
        if self.use_minibatch:
            f = self.minibatch

        start = timeit.default_timer()

        # run the algorithm
        results = [f(X) for _ in range(self.n_starts)]
        best = min(results, key=lambda x: x["dist"])

        elapsed = timeit.default_timer() - start

        self.centroids = best["centroids"]
        self.labels_ = best["z"]

        # calculate avg time and avg iteration count for each run
        self.avg_iterations = sum([x["iters"] for x in results]) / float(self.n_starts)
        self.avg_time = elapsed / float(self.n_starts)

    def initialize(self, X):
        """
        initialize the centroids used in Kmeans.
        """
        
        if self.pp:
            # k-means++
            # ----------------------------------------
            # 1. Take one center c1 uniformly from X
            # 2. Take a new center ci, choosing x from X with
            #    prob D(x)^2/sum(D(x)^2 for all x in X)
            #    where D is the shortest distance to a center we
            #    have already chosen
            # 3. Repeat 2 until we have k centers
            def D(X, centers):
                dists = [np.sqrt(np.sum((X - center)**2, axis=1)) for center in centers]
                return np.min(np.array(dists), axis=0)

            centers = []
            centers.append(X[np.random.choice(1, self.n_clusters)][0])

            for k in range(1, self.n_clusters):
                dists = D(X, centers)
                # calculate the probability mentioned above
                probs = (dists ** 2) / np.sum(dists ** 2)

                # we us cumulative sum here since we our draw the threshold uniformly
                r = np.random.rand()
                for idx, prob in enumerate(np.cumsum(probs)):
                    if r < prob:
                        centers.append(X[idx])
                        break

            return np.array(centers)
        else:
            # k-means (standard)
            # draw k random data points
            N = X.shape[0]
            return X[np.random.choice(N, self.n_clusters)]
    
    def update(self, X, z):
        """
        N, D = X.shape
        centroids = np.zeros((self.n_clusters, D))
        for i in xrange(self.n_clusters):
            # get data points in cluster i
            Xi = X[z == i]
            if Xi.shape[0] == 0:
                continue
            centroids[i] = np.mean(Xi, axis=0)
        return centroids
        """
        # if some cluster does not contain any data points, the
        # mean function will try to divide by zero. The above function
        # was replaced by the "C =" update rule below to speed things up.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            C = [X[z == k].mean(axis=0) for k in range(self.n_clusters)]
            return np.nan_to_num(C)

    def compute_distortion(self, X, centroids, z):
        distortion = 0
        for k in xrange(self.n_clusters):
            distortion += np.sum(np.sum((X[z == k] - centroids[k])**2, axis=1))
        return distortion

def assign_labels(f, k, y):
    """
    assign label to each cluster in f, return an array of labels.
    a label is assigned by majority vote, i.e. find out which true
    label occurs the most in a cluster and assign that label
    """
    labels = [-1] * k
    z = f.labels_

    for i in xrange(k):
        lbls, counts = np.unique(y[z==i], return_counts=True)
        labels[i] = lbls[np.argmax(counts)]
        
    return labels

def mistake_rate(f, k, y):
    """ evaluate (return mistake rate) for algorithm f using y (true labels) """
    cluster_labels = assign_labels(f, k, y)
    yhat = np.array([cluster_labels[yh] for yh in f.labels_])
    return (np.sum(yhat != y)/float(len(y)))

def evaluate_kmeans():
    """ run kmeans implementation on dataset """
    import dataset
    from sklearn.feature_extraction.text import TfidfVectorizer

    def kmeans_args(k):
        return {
            "KM": { "n_clusters": k, "init": "k-means", "minibatch": False },
            "KM++": { "n_clusters": k, "init": "k-means", "minibatch": False },
            "MBKM": { "n_clusters": k, "init": "k-means++", "minibatch": True },
            "MBKM++": { "n_clusters": k, "init": "k-means++", "minibatch": True },
        }

    # get all unique categories in the dataset and shuffle the order
    labels, _ = dataset.get(subset="all")
    categories = np.unique(labels)
    np.random.shuffle(categories)

    print categories

    names = ["K"]
    for name in kmeans_args(2):
        names.append(name + " (mr)")
        names.append(name + " (time)")
        names.append(name + " (it)")
    
    print ", ".join(names)
    for k in range(2,21):
        n = 0
        args = kmeans_args(k)

        # select k first categories from the list of all categories we
        # prepared above
        y, Xdata = dataset.get(categories=categories[:k], subset="all")
        vec = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2,
                                stop_words="english", use_idf=True)
        X = vec.fit_transform(Xdata)

        print "{},".format(k),
        for name in args:
            n += 1
            km = KMeans(**args[name])
            km.fit(X)

            mr = mistake_rate(km, k, y)
            time = km.avg_time
            iters = km.avg_iterations

            str = "{:.2f}, {:.2f}, {:.2f}".format(mr, time, iters)
            if n != len(args):
                str += ","
            print str,
            sys.stdout.flush()
        print ""

if __name__ == "__main__":
    if len(sys.argv) == 1:
        evaluate_kmeans()
    elif len(sys.argv) == 3 and sys.argv[1] == "plot":
        # user friendly names
        def get_name(val):
            names = {
                "KM": "KMeans",
                "KM++": "KMeans++",
                "MBKM": "MiniBatch KMeans",
                "MBKM++": "MiniBatch KMeans++"
            }
            return names[val.split(" ")[0]]
        # find out how many lines to skip
        skip = 0
        headers = None
        with open(sys.argv[2]) as f:
            for line in f:
                skip += 1
                if line[0] == "K":
                    headers = [x.strip() for x in line.split(",")]
                    break

        data = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=skip).transpose()
        Xaxis = data[0]
        # plot time on y axis
        plt.figure()
        for i in [i for i in range(len(headers)) if "(time)" in headers[i]]:
            plt.plot(Xaxis, data[i], label=get_name(headers[i]))
        plt.legend(loc = 0)
        plt.title("KMeans Computation time")
        plt.xlabel("Number of clusters / categories")
        plt.ylabel("Avg. time (s) per run")

        # plot mistake rate on y axis
        plt.figure()
        for i in [i for i in range(len(headers)) if "(mr)" in headers[i]]:
            plt.plot(Xaxis, data[i], label=get_name(headers[i]))
        plt.legend(loc = 0)
        plt.title("KMeans Mistake rate")
        plt.xlabel("Number of clusters / categories")
        plt.ylabel("Mistake rate")

        # plot avg iterations on y axis (only for kmeans / kmeans++)
        plt.figure()
        for i in [i for i in range(len(headers)) if "(it)" in headers[i] and "MB" not in headers[i]]:
            plt.plot(Xaxis, data[i], label=get_name(headers[i]))
        plt.legend(loc = 0)
        plt.title("KMeans Average iterations")
        plt.xlabel("Number of clusters / categories")
        plt.ylabel("Average iterations per run")        

        plt.show()        
    else:
        print "Usage: {} [plot CSVFILE]".format(sys.argv[0])
