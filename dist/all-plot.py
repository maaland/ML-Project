import pandas as pd
import sys
import matplotlib.pyplot as plt

'''
Assumes that Goodness results of Birch, HAC, K-means and MBK-means are stored in separate .csv files in a results directory
Plots V-measure vs. number of clusters for each of the 4 algorithms.


'''






def main():
	data = {"Birch": pd.read_csv('results/birch.csv'), "HAC": pd.read_csv('results/hac.csv'), 
	"K-Means": pd.read_csv('results/kmeans.csv'), "Mini-batch K-Means": pd.read_csv('results/mbkmeans.csv')}
	plt.figure()
	for name, d in data.iteritems(): 
	    columns = list(d.columns.values)
	    num_clusters = d['numc'].values
	    i = 1
	    l1, = plt.plot(num_clusters, d['v-meas'].values, '-', linewidth=.5, label=name)	
	
	plt.title("V-measure")
	plt.xlabel("Number of clusters")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()
