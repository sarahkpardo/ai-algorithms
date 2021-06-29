import numpy as np
np.warnings.filterwarnings('ignore', 'overflow')

class KNN:
	"""
	Class implementing K-Nearest Neighbor model
	"""
	def __init__(self, k):
		self.k = k
		self.train_pts = None
		self.train_labels = None

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def knn(self, x):
		"""
		Compute k nearest neighbors from a sample x to saved training points
		:param x: (d x 1) sample, where d == dimension of training points
		:return: (1 x k) vector of indices of nearest neighbors in training set
		"""
		n = self.train_pts.shape[0]
		assert x.shape[0] == self.train_pts.shape[1]

		diffs = np.zeros((self.n))
		for i in range(self.n):
			diff = x - self.train_pts[i, :]
			diffs[i] = diff.dot(diff)
		dists = np.sqrt(diffs)
		sort_idx = np.argsort(dists)
		return sort_idx[:self.k] # k indices into the training set

	def train(self, X, y):
		"""
		Save a set of labeled training points
		:param X: (n x d) matrix of training points
		:param y: (n x 1) vector of labels
		:return: None
		"""
		self.train_pts = X
		self.train_labels = y
		self.n = X.shape[0]

	def predict(self, X):
		"""
		Predict the class labels for of a set of test input points
		:param X: (n x d) matrix of test points
		:return: (n x 1) vector of predicted labels
		"""
		n = X.shape[0]
		d = X.shape[1]

		neighbors = np.zeros((n, self.k))  # indices of knn for n in points
		labels = np.zeros((n, self.k), dtype=np.int32)
		votes = np.zeros((n, 1))

		for i in range(n):
		    # match up data point i with all the saved training points
			# which have a label already
			neighbors[i] = self.knn(X[i])

			# use the k nearest neighbors of already-labeled points
			# to majority-vote for a label for the newly added point
			for j in range(self.k):
				labels[i, j] = self.train_labels[int(neighbors[i, j])]
			# (add new points to labeled set?)

			# Use majority voting to determine the class.
			vote = np.bincount(labels[i, :])

			# In case of a tie, pick a class at random.
			if vote.shape == (2,) and vote[0] == vote[1]:
				print('tie')
				if np.random.rand() < .5:
					votes[i] = 0
				else:
					votes[i] = 1
			elif vote.shape == (2,) and vote[0] != vote[1]:
				votes[i] = np.argmax(vote)
			elif vote.shape == (1,):
				votes[i] = 0

		return votes.squeeze()

