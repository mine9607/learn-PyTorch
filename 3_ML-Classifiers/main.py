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

iris = datasets.load_iris()

# instantiate the Standard Scaler class to standardize features (0 to 1)
sc = StandardScaler()
ppn = Perceptron(eta0=0.01, random_state=1)

# Select only features from columns at index 2 and 3
X = iris.data[:, [2, 3]]  # extract all rows and the 2nd and 3rd columns
# true label array
y = iris.target

print("Class Labels:", np.unique(y))
# Class Labels [0 1 2]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# stratify=y ensures that the train_test_split method returns training and test subsets that have the same proportions of class labels as the input dataset.

print("Labels counts in y:", np.bincount(y))
print("Labels counts in y_train:", np.bincount(y_train))
print("Labels counts in y_test:", np.bincount(y_test))

# Call the fit method from the  StandardScaler class to find the mean and std dev needed to standardize values
sc.fit(X_train)

# Standardize the feature columns
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Fit perceptron to standardized training data
ppn.fit(X_train_std, y_train)
# Make inference on test data
y_pred = ppn.predict(X_test_std)

print("Misclassified examples: %d" % (y_test != y_pred).sum())

print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
print("Accuracy: %.3f" % ppn.score(X_test_std, y_test))


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(
    X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150)
)

plt.xlabel("Petal length [standardized]")
plt.ylabel("Petal width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
