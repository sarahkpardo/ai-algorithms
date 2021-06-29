import numpy as np
np.warnings.filterwarnings('ignore', 'overflow')


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
				self.w[j, i] -= self.lr * (self.a[i, :].sum() / self.n) * 							(delta_in[j, :].sum() / self.n)
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
