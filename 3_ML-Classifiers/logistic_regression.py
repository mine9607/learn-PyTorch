import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("../2_Simple-Classification"))
from utils import plot_decision_regions


class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.

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
        self : Instance of LogisticRegressionGD
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

            # Loss function
            loss = (
                -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[0]
            )
            # Track the loss per EPOCH for plotting loss improvement per EPOCH
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # Calculate the net input z = w_1*x_1 + ... w_n * x_n + b = w^Tx + b
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """Compute linear activation"""
        # Simply return the net_input as the result of the decision function since it is linear
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        # Adaline returns a probability of a class match float (continuous) rather than 0 or 1 int like perceptron (step)
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# Load dataset
iris = datasets.load_iris()

# Filter data set to only two features
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Instantiate Standard Scalar class object
sc = StandardScaler()
sc.fit(X_train)

# Standardize the input data
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create Training Subsets
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

# Instantiate Logistic Regression Class Object
lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)

lrgd.fit(X_train_01_subset, y_train_01_subset)

plot_decision_regions(X_train_01_subset, y_train_01_subset, classifier=lrgd)

plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
