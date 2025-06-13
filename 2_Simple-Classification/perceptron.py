import numpy as np


class Perceptron:
    """Perceptron Classifier.

    Parameters
    ----------
    eta: float Learning Rate (between 0.0 and 1.0)
    n_iter: int Passes over the training dataset.
    random_state: int Random number generator seed for random weight initialization

    Attributes
    ----------
    w_ : 1d-array Weights after fitting.
    b_ : Scalar Bias unit after fitting.

    errors_: list Number of misclassifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta  # learning rate
        self.n_iter = n_iter  # number of epochs
        self.random_state = random_state  # random seed for reproducibility

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
                                  [   rows  ,      cols  ]
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of examples and n_features is the number of features

        y : array-like, shape = [n_examples]
        Target values.

        Returns
        ------
        self: Object
        """

        # Create a random number generator (can use with any statistical distribution)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # loc = mean of distribution
        # scale = std dev of distribution
        # size = output of the distribution--in this case we will output a 1-D array-like of randomized weights (1 per feature)
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            # zip creates a tuple of the two data sources (must have the same number of examples or will truncate to shorter dataset)
            for xi, target in zip(X, y):
                # calculate the update parameter for weight and bias
                update = self.eta * (target - self.predict(xi))
                # calculate the weight for a each feature and the bias
                self.w_ += update * xi
                self.b_ += update
                # Calculate the error and append to the list of errors for each weight or feature
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    # USED AFTER TRAINING - helper function to apply the learned weights and bias from the FIT method to incoming test examples
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    # Creating the binary classification--np.where() checks the condition and if TRUE (>=0) returns 1 else returns 0
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
