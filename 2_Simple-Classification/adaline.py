import numpy as np


class Adaline:
    """ADAptive LInear NEuron Classifier

    Parameters
    ---
    eta : float
        Learning rate (between 0.0 and 1.0)

    n_iter : int
        Passes over the trianing dataset (epochs)

    random_state : int
        Random number generator seed for random weight initializations


    Attributes
    ---
    w_ : 1d-array
        Weights after fitting
    b_ : Scalar
        Bias unit after testing
    losses_ : list
        Mean squared error loss function values in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data

        Parameters
        ---
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples  and n_features is the number of features

        y : {array-like}, shape = [n_examples]
            Target values (labels or classes)

        Returns
        ---
        self : Object
        """

        # set the random seed in numpy
        rgen = np.random.RandomState(self.random_state)
        # 1D array of weights (1 per feature)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # scalar single value of bias term
        self.b_ = np.float64(0.0)
        # array of loss values per example: true label - predicted label
        self.losses_ = []

        # for each EPOCH
        for i in range(self.n_iter):
            # net_input = sum(w_i * X_i) + b
            net_input = self.net_input(X)

            # linear activation function (not step like with perceptron)::: sigma(z) = z
            output = self.activation(net_input)
            # true label - predicted label
            errors = y - output
            # Partial derivative of weights matrix for gradient descent
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            # Partial derivative of bias term for gradient descent
            self.b_ += self.eta * 2.0 * errors.mean()
            # Loss function (Mean Square Error) from ALL VALUES, EACH EPOCH
            loss = (errors**2).mean()
            # Track the loss per EPOCH for plotting loss improvement per EPOCH
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # Calculate the net input z = w_1*x_1 + ... w_n * x_n + b = w^Tx + b
        return np.dot(X, self.w_) + self.b_

    def activation(self, net_input):
        """Compute linear activation"""
        # Simply return the net_input as the result of the decision function since it is linear
        return net_input

    def predict(self, X):
        """Return class label after unit step"""
        # Adaline returns a probability of a class match float (continuous) rather than 0 or 1 int like perceptron (step)
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class AdalineSGD:
    """ADAptive LInear NEuron Classifier

    Parameters
    ---
    eta : float
        Learning rate (between 0.0 and 1.0)

    n_iter : int
        Passes over the trianing dataset

    shuffle : bool
        Shuffles the training data every epoch if True to prevent cycles

    random_state : int
        Random number generator seed for random weight initializations


    Attributes
    ---
    w_ : 1d-array
        Weights after fitting
    b_ : Scalar
        Bias unit after testing
    losses_ : list
        Mean squared error loss function value averaged over all training examples in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data

        Parameters
        ---
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples  and n_features is the number of features

        y : {array-like}, shape = [n_examples]
            Target values (labels or classes)

        Returns
        ---
        self : Object
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)

        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.0, size=m)
        self.b_ = np.float64(0.0)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
