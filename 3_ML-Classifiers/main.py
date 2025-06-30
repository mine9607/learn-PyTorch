import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

sc = StandardScaler()
ppn = Perceptron(eta0=0.01, random_state=1)

X = iris.data[:, [2, 3]]  # extract all rows and the 2nd and 3rd columns
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

sc.fit(X_train)  # estimates the sample mean and deviation for each feature dimension
X_train_std = sc.transform(X_train)  # standardizes the data with the mean and std dev
X_test_std = sc.transform(X_test)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print("Misclassified examples: %d" % (y_test != y_pred).sum())
