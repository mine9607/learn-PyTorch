import numpy as np


class Adaline:
    """ADAptive LInear NEuron Classifier

    Parameters
    ---
    eta : float
        Learning rate (between 0.0 and 1.0)

    n_iter : int
        Passes over the trianing dataset

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
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.losses_ = []

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
