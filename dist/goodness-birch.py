from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import Birch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from time import time
import dataset


#print "Loading features"

def make_string_label_dict(unique_string_labels):
	label_dict = dict()
	for i in range(unique_string_labels.size):
		label_dict[unique_string_labels[i]] = i
	return label_dict

# trunc_label specifies whether to truncate the label
# to the least common denominator for each usenet group
labels, data = dataset.get( truncate_label=False)
datapoints = len(data)
#print "Number of datapoints: ", datapoints
unique_labels,_ = np.unique(labels, return_inverse=True)
#print "- Labels:", unique_labels
# Create a dictionary with enumerated label names
label_dict = make_string_label_dict(unique_labels)
#print label_dict
# true_k holds the true number of clusters
true_k = np.unique(labels).shape[0]

# Calculate and print metrics to assess k-means
def assess_birch(estimator, num_clusters, data, labels):
	t0 = time()
	estimator.fit(data)
	print ", ".join(map(str, [num_clusters, metrics.homogeneity_score(labels, estimator.labels_),metrics.completeness_score(labels, estimator.labels_),metrics.v_measure_score(labels, estimator.labels_), calculate_mistake_rate(estimator, num_clusters, data, labels)])) 
	'''print "%d 		%.2fs 	%i    %.3f    %.3f    %.3f    %.3f    %.3f   %.3f" % (num_clusters, (time()-t0), estimator.inertia_, metrics.homogeneity_score(labels, estimator.labels_),
		metrics.completeness_score(labels, estimator.labels_),
		metrics.v_measure_score(labels, estimator.labels_),
		metrics.adjusted_rand_score(labels, estimator.labels_),
		metrics.adjusted_mutual_info_score(labels, estimator.labels_),
		calculate_mistake_rate(estimator, num_clusters, data, labels))
	'''
	

def calculate_mistake_rate(estimator, num_clusters, data, labels):
	cluster_assignments = estimator.predict(data)
	mistakes = 0.0
	for i in range(num_clusters):
		cluster_datapoints = []
		for n in range(cluster_assignments.size):
			if i==cluster_assignments[n]:
				cluster_datapoints.append(n)
		# Datapoints belonging to cluster i
		cluster_datapoints = np.array(cluster_datapoints)
		# Gets the string labels of the datapoints in cluster i
		true_labels = []
		for l in cluster_datapoints:
			true_labels.append(label_dict[labels[l]])
		true_labels = np.array(true_labels)
		_, p = np.unique(true_labels, return_inverse=True)
		# Counts the number of each index
		counts = np.bincount(p)
		# Gets the index with the highest count
		maxpos = counts.argmax()
		mistakes += (p!=maxpos).sum()
	return mistakes/datapoints

			





#print "Preparing Tfidf vectorizer"

# prepare features
vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)

X = vectorizer.fit_transform(data)
#print "Fitting K-means for clusters 1 through 20"
#print "___________________"
#print '% 9s' % 'clusters	time 	inertia	   homo    compl    v-meas    ARI    AMI    Mistake Rate'
print "numc,homo,comp,v-meas,mr"
for numc in range(1, true_k+1):
	birch = Birch(n_clusters=numc)
	assess_birch(birch, numc, X, labels)


