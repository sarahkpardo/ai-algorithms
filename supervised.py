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


class MLP:
	"""
	Implement multi-layer perceptron model with sigmoid activation and
	fully-connected hidden layer

	n.b., my understanding of this is primarily based on + structured by
	http://neuralnetworksanddeeplearning.com/chap2.html

	"""
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		"""
		Compute mean squared error between prediction and target
		:param prediction: output of final layer
		:param target: value to be approximated by output
		:return: mean squared error between prediction and target
		"""
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		"""
		Partial derivative of MSE w/r/t output of last layer
		:param prediction: output of final layer
		:param target: value to be approximated by output
		:return: partial derivative of MSE w/r/t output of last layer
		"""
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			#if s % 5000 == 0:
			#	print('MLP step:', s)
			i = s % y.size
			if (i == 0):
				X, y = self.shuffle(X, y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)


class FCLayer:
	"""
	Class for fully connected NN layer
	"""
	def __init__(self, w, b, lr=.001):
		"""
		Save tunable parameters
		n.b.: I am transposing w and b because it makes more sense to me
		mathematically to treat the layers as applying the weight matrix
		transform to column vectors, which then also makes the partial
		derivatives more intuitive. However it is at the expense of
		having to transpose when the taking input to a layer and transposing
		it back so the output for the next layer is shaped as expected by
		the structure of the training loop

		:param w: (input features x output features) matrix of weights
		:param b: (1 x output features) offset vector
		:param lr: learning rate
		"""
		self.lr = lr

		self.w = w.T # (out x in)
		self.b = b.T # (out x 1)

		self.in_d = w.shape[0]
		self.out_d = w.shape[1]

	def forward(self, input):
		"""
		Apply forward pass to input
		:param input: (num. samples x input features) input batch
		:return: (num. samples x output features) linearly transformed output batch
		"""
		# input = (n x in)
		assert input.shape[1] == self.in_d
		self.n = input.shape[0]
		# a = (in x n)
		self.a = input.T

		# self.z: (out x n) weighted input to activation l
		# (out x n) = (out x in).dot(in x n) + (out x 1)
		self.z = np.dot(self.w, self.a) + self.b

		# assert output shape == (num. samples x output features)
		# output = self.z.T
		#assert output.shape[1] == self.out_d

		return self.z.T

	def backward(self, gradients):
		"""
		Apply backpropagation to propagate error through the computation graph;
		update weights
		:param gradients: (out_d x 1) delta vector from layer l+1 (delta_l+1)
		:return: W^T.dot(delta_l+1)
		"""
		# transpose gradients to be (in x n)
		delta_in = gradients.T
		assert delta_in.shape[0] == self.out_d
		# delta.shape = (in x n) = (in x out).dot(out x n)
		delta_out = self.w.T.dot(delta_in)
		#assert delta_out.shape[0] == self.in_d

		# update weights
		for i in range(self.in_d):
			for j in range(self.out_d):
				# w = (j x i)
				# delta_in = (j x n)
				# a(ctivation) = (i x n)
				# \partial C / \partial W_(j, i) = a_i * delta_j
				# take average over samples if there are multiple
				self.w[j, i] -= self.lr * (self.a[i, :].sum() / self.n) * (delta_in[j, :].sum() / self.n)
				self.b[j] -= self.lr * delta_in[j, :].sum() / self.n

		return delta_out.T # (n x in_d)

class Sigmoid:
	"""
	Sigmoid activation l applied to layer l
	"""
	def __init__(self):
		self.sigmoid = lambda x: np.reciprocal(np.add(1, np.exp(x * -1)))

	def forward(self, input):
		"""
		Apply sigmoid nonlinearity element-wise
		:param input: (num. samples x input features) batch z_l
		:return: out_ij = 1 / (1 + exp(-z_ij)
		"""
		self.z = input.T
		self.in_d = self.z.shape[0]
		return self.sigmoid(input)

	def backward(self, gradients):
		"""
		Update gradients for sigmoid activation of layer l
		:param gradients: (W_l+1).T.dot(delta_l+1)
		:return: delta_l = gradients (hadamard) sigmoid'(z_l)
		"""
		assert gradients.shape[1] == self.in_d
		sig_deriv = self.sigmoid(self.z) * np.add(1, -1 * self.sigmoid(self.z))
		return np.multiply(gradients, sig_deriv)