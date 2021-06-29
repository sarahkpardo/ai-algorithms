import math

import numpy as np

class AGNES:
	"""
	Class implementing single-link AGNES clustering, where the distance
	between cluster a and b is defined as the Euclidean distance between
	closest members of clusters a and b.
	"""
	def __init__(self, k):
		"""
		:param k: number of clusters
		"""
		self.k = k

	def distance(self, a, b):
		"""
		Compute distance between points a and b
		:return: Euclidean distance
		"""
		diffs = (a - b) ** 2
		return np.sqrt(diffs.sum())

	def train(self, X):
		"""
		Classify a dataset using AGNES clustering implemented with Kruskal's
		algorithm for finding minimum spanning trees.
		Terminates when the number of clusters is equal to k (produces a
		forest of k trees). Assumes all data is continuous.

		:param X: (n x d) numpy array of unlabeled points
		:return: (n x 1) numpy array with cluster ids for each of n points in X
		"""
		n = X.shape[0]
		d = X.shape[1]
		y = np.zeros(n)

		# Populate dissimilarity graph
		D = []
		for i in range(n):
			for j in range(i + 1, n):
				D.append([i, j, ((X[i] - X[j]) ** 2).sum()])

		# Sort D by edge weight (distances)
		D = sorted(D, key=lambda item: item[2])

		# Union by weight with path compression
		# https://www2.cs.duke.edu/courses/cps100e/fall09/notes/UnionFind.pdf

		# Trace parent indices to find roots of clusters
		idx = []
		# Weight (number of elements in cluster)
		weight = []

		# Start with n clusters of single elements
		for node in range(n):
			idx.append(node)
			weight.append(1)

		# Number of edges to be added is equal to n - k (k = number of clusters)
		# -> (n - 1) edges in fully connected MST
		# -> add (k - 1) edges to add to merge k clusters into (k - 1) clusters
		# -> (n - 1) - (k - 1)
		edge = 0
		i = 0
		while edge < n - self.k:
			# Process the next smallest edge
			u, v, dist = D[i]
			i += 1

			# Find the cluster root of each of the edge endpoints u, v
			uRoot = u
			while (uRoot != idx[uRoot]):
				# Faster traversal: halve the path length by making every other
				# node in path point to its grandparent
				idx[uRoot] = idx[idx[uRoot]]
				uRoot = idx[uRoot]
			vRoot = v
			while (vRoot != idx[vRoot]):
				idx[vRoot] = idx[idx[vRoot]]
				vRoot = idx[vRoot]

			# If endpoints are not in the same cluster, join their clusters
			# with union by weight for disjoint sets
			if uRoot != vRoot:
				edge += 1
				uRoot = u
				while (uRoot != idx[uRoot]):
					idx[uRoot] = idx[idx[uRoot]]
					uRoot = idx[uRoot]
				vRoot = v
				while (vRoot != idx[vRoot]):
					idx[vRoot] = idx[idx[vRoot]]
					vRoot = idx[vRoot]

				# Add smaller tree to larger tree
				if weight[uRoot] < weight[vRoot]:
					idx[uRoot] = vRoot
					weight[vRoot] += weight[uRoot]
				else:
					idx[vRoot] = uRoot
					weight[uRoot] += weight[vRoot]

		# Retrieve the weight of the root point for each point in X
		# Different weights distinguish the clusters
		for point in range(n):
			root = point
			while (root != idx[root]):
				idx[root] = idx[idx[root]]
				root = idx[root]
			y[point] = weight[root]

		return y
	
