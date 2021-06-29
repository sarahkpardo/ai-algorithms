import numpy as np
np.warnings.filterwarnings('ignore', 'overflow')

class Perceptron:
	def __init__(self, w, b, lr):
		"""
		Class implementing the Perceptron model
		:param w: (dim. input x 1) vector of model weights
		:param b: (1 x 1) bias term
		:param lr: Learning rate
		"""
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		"""
		Fit weights and bias to a given set of labeled data points
		:param X: (n x d) matrix of data points
		:param y: (1 x d) vector of labels
		:param steps: Number to update steps to perform
		:return: None
		"""
		n = X.shape[0]
		self.d = X.shape[1]

		#for full-batch updates:
		#W = np.ones(self.d + 1)
		#W[0] = self.b
		#W[1:] = self.w
		#Xprime = np.ones((n, self.d + 1))
		#Xprime[:, 1:] = X


		for _ in range(steps):
			#if _ % 5000 == 0:
			#	print(_)
			i = _ % n
			if (i == 0):
				idxs = np.arange(y.size)
				np.random.shuffle(idxs)
				X, y = X[idxs], y[idxs]

			# For each step update the model on a single datapoint
			# (stochastic updates)
			y_hat = np.where((np.dot(self.w, X[i]) + self.b) >= 0.0, 1, 0)
			update = self.lr * (y[i] - y_hat)
			self.w += update * X[i]
			self.b += update

			# OR apply on the entire batch at once (non-stochastic update)
			#y_hat = np.where(np.dot(Xprime, W) <= 0.0, 0, 1)
			#update = self.lr * (y - y_hat)
			#W += update.dot(Xprime)

		#self.w = W[1:]
		#self.b = W[0]

		return None

	def predict(self, X):
		"""
		Apply model to an input data point to return a predicted label
		:param X: (n x d) matrix of input data points
		:return: (n x 1) vector of predicted labels
		"""
		n = X.shape[0]
		assert X.shape[1] == self.d

		preds = np.zeros(n)
		for i in range(n):
			preds[i] = np.where(np.dot(self.w, X[i]) + self.b <= 0.0, 0, 1)
		return preds.squeeze()


