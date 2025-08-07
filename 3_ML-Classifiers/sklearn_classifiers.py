import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath("../2_Simple-Classification"))
from utils import plot_decision_regions

ppn = Perceptron(eta0=0.01, random_state=1)
lr = LogisticRegression(C=100.0, solver="lbfgs", multi_class="ovr")
svm = SVC(kernel="linear", C=1.0, random_state=1)
svc1 = SVC(kernel="rbf", random_state=1, gamma=10, C=1.0)
svc2 = SVC(kernel="rbf", random_state=1, gamma=100.0, C=1.0)
dt = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
knn1 = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
knn2 = KNeighborsClassifier(n_neighbors=10, p=2, metric="minkowski")

ppn2 = SGDClassifier(loss="perceptron")
lr2 = SGDClassifier(loss="log")
svm2 = SGDClassifier(loss="hinge")

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, test_size=0.3, stratify=y
)

# Standardize features
sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Fit Perceptron Model
ppn.fit(X_train_std, y_train)

# Fit SVM Model
svm.fit(X_train_std, y_train)

# Fit SVC Models
svc1.fit(X_train_std, y_train)
svc2.fit(X_train_std, y_train)

# Fit Decision Tree
dt.fit(X_train_std, y_train)

# Fit Random Forest Classifier
forest.fit(X_train_std, y_train)

# Fit K Neighbors
knn1.fit(X_train_std, y_train)
knn2.fit(X_train_std, y_train)

# Fit Logistric Regression Model
lr.fit(X_train_std, y_train)
print(lr.predict_proba(X_test_std[:3, :]), "\n")
print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1), "\n")
print(lr.predict(X_test_std[:3, :]), "\n")
print(lr.predict(X_test_std[0, :].reshape(1, -1)), "\n")

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# Plot Canvas
fig, axes = plt.subplots(3, 3, figsize=(12, 6))
axes = axes.flatten()

models = [svm, svc1, svc2, ppn, lr, dt, forest, knn1, knn2]
titles = [
    "SVM Linear",
    "SVC gamma=10",
    "SVC gamma=100",
    "Perceptron",
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "K-Nearest Neighbors Classifier k=5",
    "K-Nearest Neighbors Classifier k=10",
]

for ax, model, title in zip(axes, models, titles):
    plot_decision_regions(
        X_combined_std, y_combined, classifier=model, test_idx=range(105, 150), ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Petal length [standardized]")
    ax.set_ylabel("Petal width [standardized]")

plt.tight_layout()
plt.show()

# feature_names = ["Sepal Length", "Sepal Width", "Petal length", "Petal width"]
feature_names = ["Petal length", "Petal width"]
tree.plot_tree(dt, feature_names=feature_names, filled=True)
plt.show()
