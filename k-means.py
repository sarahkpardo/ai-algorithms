import math

import numpy as np


class K_MEANS:
	"""
	Class implementing k-means clustering using Euclidean distance between
	features.
	"""

	def __init__(self, k, t):
		"""
		:param k: number of clusters
		:param t: max number of iterations
		"""
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		"""
		Compute sum of squared errors distance from data point to
		centroids.
		"""
		diffs = (centroids - datapoint) ** 2
		return np.sqrt(diffs.sum(axis=1))

	def train(self, X):
		"""
		Assumes all data is continuous. Ties are broken randomly.

		:param X: (n x d) numpy array of unlabeled data points
		:return: (n x 1) numpy array of cluster ids for each of n points in X
		"""
		# number of elements
		n = X.shape[0]
		# dimension of data points
		d = X.shape[1]
		# array for labels
		y = np.full(n, 0)

		# initialize with random centroids
		r = np.random.permutation(n)[:self.k]
		centroids = X[r]

		for _ in range(self.t):
			# expectation: assign each point to its closest centroid
			# assign point x to argmin of [(c1 - x) (c2 - x) ... (ck - x)]
			for i in range(n):
				x = X[i]
				dists = np.zeros(3)
				for j in range(self.k):
					dists[j] = np.sqrt(((centroids[j] - x) ** 2).sum())
				y[i] = np.argmin(dists)

			# maximization: compute the new centroid (mean) of each cluster
			for i in range(self.k):
				x = X[y == i]
				mean = x.sum(axis=0) / len(x)
				centroids[i] = mean

		return y

	
