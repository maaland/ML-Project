from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.externals import joblib
from sklearn.neighbors import kneighbors_graph
import numpy as np
from time import time
import dataset


def make_string_label_dict(unique_string_labels):
	label_dict = dict()
	for i in range(unique_string_labels.size):
		label_dict[unique_string_labels[i]] = i
	return label_dict

# trunc_label specifies whether to truncate the label
# to the least common denominator for each usenet group
labels, data = dataset.get(truncate_label=False)
datapoints = len(data)
#print "Number of datapoints: ", datapoints
labels=np.array(labels)
unique_labels,_ = np.unique(labels, return_inverse=True)
#print "- Labels:", unique_labels
# Create a dictionary with enumerated label names
label_dict = make_string_label_dict(unique_labels)
#print label_dict
# true_k holds the true number of clusters
true_k = np.unique(labels).shape[0]

# Calculate and print metrics to assess spectral clustering
def assess_hac(estimator, num_clusters, data, labels):
	t0 = time()
	cluster_assignments = estimator.fit_predict(data)
	print ", ".join(map(str, [num_clusters, metrics.homogeneity_score(labels, estimator.labels_),metrics.completeness_score(labels, estimator.labels_),metrics.v_measure_score(labels, estimator.labels_), calculate_mistake_rate(estimator, num_clusters, data, cluster_assignments, labels)])) 

'''	print "%d 		%.2fs     %.3f    %.3f    %.3f    %.3f    %.3f   %.3f" % (num_clusters, (time()-t0), metrics.homogeneity_score(labels, estimator.labels_),
		metrics.completeness_score(labels, estimator.labels_),
		metrics.v_measure_score(labels, estimator.labels_),
		metrics.adjusted_rand_score(labels, estimator.labels_),
		metrics.adjusted_mutual_info_score(labels, estimator.labels_),
		calculate_mistake_rate(estimator, num_clusters, data, cluster_assignments, labels))
'''

def calculate_mistake_rate(estimator, num_clusters, data, cluster_assignments, labels):
	cluster_assignments = cluster_assignments
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


#print "Fitting HAC for clusters 1 through 20"
#print "___________________"
#print '% 9s' % 'clusters	time	 homo    compl    v-meas    ARI    AMI    Mistake Rate'
print "numc,homo,comp,v-meas,mr"
for numc in range(1, true_k+1):
	hac = AgglomerativeClustering(n_clusters=numc, linkage='ward')
	assess_hac(hac, numc, X.toarray(), labels)

