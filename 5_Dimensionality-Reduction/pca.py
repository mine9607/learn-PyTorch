import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import newaxis, random
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("../2_Simple-Classification"))
from utils import plot_decision_regions

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
    "0D280/0D315 of diluted wines",
    "Proline",
]

# Step 1 - Standardize the Dataset
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Step 2 - Create Covariance Matrix

cov_mat = np.cov(
    X_train_std.T
)  # computes the covariance matrix of the standardized training data-set
print(cov_mat)
print(cov_mat.shape)

# Step 3 - Obtain the eigenvalues and eigend vectors
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("\nEigenvalues: \n", eigen_vals)

# Step 4 - Sort the eigenvalues by decreasing order to rank eigenvectors
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, align="center", label="Individual explained variance")
plt.step(range(1, 14), cum_var_exp, where="mid", label="Cumulative explained variance")
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal component index")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Step 5 - Select k eigenvectors which correspond to k largest eigenvalues (where k is the dimensionality of the new feature subspace)

# make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]

# sort the list of tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# collect the eigenvectors corresponding to largest eigenvalues to capture > 60% of the variance in the data
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
# print the 13 x 2 dimensional matrix W
print("Matrix W: \n", w)

# transform an example x using the new matrix W (i.e. 1 x 13 dot 13 x 2 = 1 x 2 matrix
test_example_transformed = X_train_std[0].dot(w)

X_train_pca = X_train_std.dot(w)

colors = ["r", "b", "g"]
markers = ["o", "s", "^"]
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(
        X_train_pca[y_train == l, 0],
        X_train_pca[y_train == l, 1],
        c=c,
        label=f"Class {l}",
        marker=m,
    )

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()


# initializing the PCA transformer and logistic regression esimator
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class="ovr", random_state=1, solver="lbfgs")

# dimensionality reduction
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

# fitting the logistic regression model on the reduced dataset:
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.title("Logistic Regression on Reduced Feature Subspace (Training Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.title("Logistic Regression on Reduced Feature Subspace (Test Set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# NOTE: we have effectively created a decent classifier using only 2 features compared to the original 13

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)

loadings = eigen_vecs * np.sqrt(eigen_vals)

fig, ax = plt.subplots()
ax.bar(range(13), loadings[:, 0], align="center")
ax.set_ylabel("Loadings for PC1")
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()

sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig, ax = plt.subplots()
ax.bar(range(13), sklearn_loadings[:, 0], align="center")
ax.set_ylabel("Sklearn Loadings for PC1")
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()

# Linear Discriminant Analysis (LDA)
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f"MV {label}: {mean_vecs[label-1]}\n")

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print("Within-class scatter matrix: " f"{S_W.shape[0]}x{S_W.shape[1]}")


# We assume that the class labels in the training dataset are uniformly distributed but if we print them they are not
print("Class label distribution:", np.bincount(y_train)[1:])

# We need to scale the individual scatter matrices S_i, before we sum them up as the scatter matrix S_w
d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print("Scaled within-class scatter matrix: " f"{S_W.shape[0]}x{S_W.shape[1]}")
