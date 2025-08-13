import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize._lsq.trf import update_tr_radius
from scipy.special import comb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def ensemble_error(n_classifier, error):
    if error <= 0.0:
        return 0.0
    if error >= 1.0:
        return 1.0

    k_start = int(math.ceil(n_classifier / 2.0))
    probs = [
        comb(n_classifier, k) * error**k * (1 - error) ** (n_classifier - k)
        for k in range(k_start, n_classifier + 2)
    ]
    return sum(probs)


print(ensemble_error(n_classifier=11, error=0.25))

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

plt.plot(error_range, ens_errors, label="Ensemble error", linewidth=2)
plt.plot(error_range, error_range, linestyle="--", label="Base error", linewidth=2)

plt.xlabel("Base error")
plt.ylabel("Base/Ensemble error")
plt.legend(loc="upper left")
plt.grid(alpha=0.5)
plt.show()

# Example of finding the ensemble label based on argmax of the weighted classifier predictions

print("Majority Vote:", np.argmax(np.bincount([0, 0, 1])))
print(
    "C3 Weighted Majority Vote:",
    np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])),
    "\n",
)

ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
print("Array of Classifier Probabilities:\n", ex)
print(ex.shape)
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
print("\nAverage Weighted Classifier Probabilities:\n", p)
print("\nPredicted Class:", np.argmax(p))

# Majority Vote Classifier


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote="classlabel", weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        if self.vote not in ("probability", "classlabel"):
            raise ValueError(
                f"vote must be 'probability' or 'classlabel'; got (vote={self.vote})"
            )
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(
                f"Number of classifiers and weights must be equal"
                f"; got len{(self.weights)} weights, {len(self.classifiers)} classifiers"
            )
        # Use LabelEncoder to ensure class labels start with 0, which is important for np.argmax call in self.predict
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote
            # Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions,
            )
        maj_vote = self.labelenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f"{name}__{key}"] = value
            return out


# Load Iris Dataset
iris = load_iris()
# Select only examples for Iris-Versicolor and Iris-virginica (only sepal width and petal length features)
X, y = iris.data[50:, [1, 2]], iris.target[50:]

# Encode the label classes to (0, 1)
le = LabelEncoder()
y = le.fit_transform(y)

# Prepare the data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1, stratify=y
)

# Define classifiers

clf1 = LogisticRegression(penalty="l2", C=0.001, solver="lbfgs", random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")

pipe1 = Pipeline([["sc", StandardScaler()], ["clf", clf1]])
pipe3 = Pipeline([["sc", StandardScaler()], ["clf", clf3]])

clf_labels = ["Logistic Regression", "Decision Tree", "KNN"]
print("10-fold cross validation:\n")

# Recall that Decision Tree Classifiers do not need scaling which is why there is no pipe2

# Estimate classifier performance (ROC AUC)
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(
        estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc"
    )
    print(f"ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]")

# Combine Classifiers into Ensemble for Majority Vote

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels += ["Majority voting"]
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(
        estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc"
    )
    print(f"ROC AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]")

# Plot the Results of Classifiers and Majority Vote
colors = ["black", "orange", "blue", "green"]
linestyles = [":", "--", "-.", "-"]
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label=f"{label} (auc = {roc_auc:.2f})")
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate (TPR)")
plt.show()

# Evaluate the Decision Boundaries of Ensemble Classifier

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(
        X_train_std[y_train == 0, 0],
        X_train_std[y_train == 0, 1],
        c="blue",
        marker="o",
        s=50,
    )
    axarr[idx[0], idx[1]].scatter(
        X_train_std[y_train == 1, 0],
        X_train_std[y_train == 1, 1],
        c="green",
        marker="o",
        s=50,
    )
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(
    -3.5, -5.0, s="Sepal width [standardized]", ha="center", va="center", fontsize=12
)
plt.text(
    -12.5,
    4.5,
    s="Petal length [standardized]",
    ha="center",
    va="center",
    fontsize=12,
    rotation=90,
)
plt.show()

# get the individual classifier parameters for optimization via grid_search

# print(mv_clf.get_params())

# Tune the inverse regularization parameter 'C' of logistic regression classifier and the depth of decision tree using grid search

params = {
    "decisiontreeclassifier__max_depth": [1, 2],
    "pipeline-1__clf__C": [0.001, 0.1, 100.0],
}

grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring="roc_auc")

grid.fit(X_train, y_train)

for r, _ in enumerate(grid.cv_results_["mean_test_score"]):
    mean_score = grid.cv_results_["mean_test_score"][r]
    std_dev = grid.cv_results_["std_test_score"][r]
    params = grid.cv_results_["params"][r]
    print(f"{mean_score:.3f} +/- {std_dev:.2f} {params}")

print(f"Best parameters: {grid.best_params_}")
print(f"ROC AUC: {grid.best_score_:.2f}")

# Applying Bagging

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)

df_wine.columns = [
    "Class Label",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

# drop 1 class - select 2 features
df_wine = df_wine[df_wine["Class Label"] != 1]
y = df_wine["Class Label"].values
X = df_wine[["Alcohol", "OD280/OD315 of diluted wines"]].values

# Encode class labels 2 and 3 to be [0,1]
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset into 80 training / 20 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Instantiate Decision Tree and Bagging Classifiers

tree = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=None)

bag = BaggingClassifier(
    base_estimator=tree,
    n_estimators=500,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    n_jobs=1,
    random_state=1,
)

# Predict Accuracy of training and test datasets to compare performance of bagging classifier to the performance of an unpruned decision tree

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f"Decision tree train/test accuracies {tree_train:.3f}/{tree_test:.3f}")

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print(f"Bagging classifier train/test accuracies {bag_train:.3f}/{bag_test:.3f}")

# Compare the decision regions between the tree classifier and the bagging classifier

x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex="col", sharey="row", figsize=(8, 3))

for idx, clf, tt in zip([0, 1], [tree, bag], ["Decision Tree", "Bagging"]):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(
        X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="blue", marker="^"
    )
    axarr[idx].scatter(
        X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="green", marker="o"
    )
    axarr[idx].set_title(tt)

axarr[0].set_ylabel("OD280/OD315 of diluted wines", fontsize=12)
plt.tight_layout()
plt.text(
    0,
    -0.2,
    s="Alcohol",
    ha="center",
    va="center",
    fontsize=12,
    transform=axarr[1].transAxes,
)
plt.show()

# Walkthrough AdaBoost

# Calculate the weighted error rate (epsilon)
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
y_hat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])

correct = y == y_hat
weights = np.full(10, 0.1)
print(weights)

# NOTE: '~' is the bitwise NOT (bitwise inversion) operator - flips all bits
epsilon = np.mean(~correct)
print(epsilon)

# Compute alpha_j coefficient
alpha_j = 0.5 * np.log((1 - epsilon) / epsilon)
print(alpha_j)

# Update the weight vector w for correctly classified examples
update_if_correct = 0.1 * np.exp(-alpha_j * 1 * 1)
print(update_if_correct)

# Update the weight vector w for misclassified examples
update_if_wrong_1 = 0.1 * np.exp(-alpha_j * 1 * -1)
print(update_if_wrong_1)

# Alternatively
# update_if_wrong_2 = 0.1 * np.exp(-alpha_j * -1 * 1)

# Update the weights
weights = np.where(correct == 1, update_if_correct, update_if_wrong_1)
print(weights)

# normalize the weights
normalized_weights = weights / np.sum(weights)
print(normalized_weights)

# Applying AdaBoost in ScikitLearn

tree = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=1)

ada = AdaBoostClassifier(
    base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1
)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f"Decision Tree train/test accuracies {tree_train:.3f}/{tree_test:.3f}")

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print(f"Decision Tree train/test accuracies {ada_train:.3f}/{ada_test:.3f}")

# Gradient Boosting
