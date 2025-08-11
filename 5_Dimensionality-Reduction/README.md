# Chapter 5

## Compressing Data via Dimensionality Reduction

### Feature Extraction

- Principal Component Analysis (PCA) (unsupervised data compression)
- Linear Discriminant Analysis (supervised dimensionality reduction - maximizes class separability)
- Dimensionality Reduction and t-distributed Stochastic Neighbor Embeeding (t-SNE) (data visualization)

Feature extraction is used to transform or project the data from a higher dimensional feature space onto a new feature space

> NOTE: Feature extraction can be thought of as a form of data compression while maintaining the most relevant information

Can improve the predictive performance of high dimensional data sets (especially if working with `non-regularized` models)

### Unsupervised Dimensionality Reduction via PCA

`PCA` - `unsupervised` linear transformation technique that is widely used for `feature extraction` and `dimensionality reduction`

Helps to identify patterns in data based on the correlation betwseen features. It aims to find the directions of maximum variance in high-dimensional data and project the data onto a new subspace of equal or fewer dimenions

> Note: The orthogonal axes (principal components) of the new subspace can be interpreted as the directions of maximum variance given the constraint that the new feature axes are orthogonal to each other

#### Steps to PCA for Dimensionality Reduction

1. create a d x k-dimensional transformation matrix (W)
2. map a vector of the features of the training example (x) onto a new k-dimensional feature subspace that has fewer dimensions than the original d-dimensional feature space

typically k << d

The first principal component will have the largest possible variance and all consequent principal components will have the largest variance given the constraint that these components are `uncorrelated` (e.g. orthogonal) to the other principal components--even if the input features **ARE** correlated, the resulting principal components will be mutually orthogonal (uncorrelated)

1. Standardize the d-dimensional dataset
2. Construct the covariance matrix
3. Decompose th covariance matrix into its eigenvectors and eigenvalues
4. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors
5. Select k eigenvectors, which correspond to the k largest eigenvalues, where k is the dimensionality of the new feature subspace (k <= d)
6. Construct a projection matrix, **W**, from the "top" k eigenvectors
7. Transform the d-dimensional input dataset, X, using the projection matrix, W, to obtain the new k-dimensional feature space

Note this is matrix multiplication (inner dimensions must match) - resulting matrix is
3x3 x 3x2 = 3 x 2

3x2 x 2x3 = 3 x 3
2x3 x 3x2 = 2 x 2

```python
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# initializing the PCA transformer and logistic regression esimator
pca = PCA(n_components=2)
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

# dimensionality reduction
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# fitting the logistic regression model on the reduced dataset:
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

print(pca.explained_variance_ratio_)

```

#### Assessing Feature Contributions

How can we assess the contributions of the original features to the principal components. How can we determine how much each original feature contributes to a given principal component?

These contributions are called `loadings`

The `factor loadings` can be computed by scaling the eigenvectors by the square root of the eigenvalues. The resulting values can then be interpreted as the correlation between the original features and the principal component.

### Linear Discrimant Analysis (LDA)

LDA is a `supervised` linear transformation technique that takes class label information into account (random forest also takes labels into account)

The goal in LDA is to find the feature subspace that optimizes class separability.

> NOTE: LDA **ASSUMES** that the data is normally distributed, the classes have identical covariance matrices and that the training examples are statistically independent of each other

#### Steps of Linear Discriminant Analysis

1. Standardize the d-dimensional dataset (d is the number of features)
2. For each class, compute the d-dimensional mean vector
3. Construct the between-class scatter matrix, $S_B$, and the within-class scatter matrix, $S_W$
4. Compute the eigenvectors and corresponding eigenvalues of the matrix, $S_W^{-1}S_B$
5. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors
6. Choose the k eigenvectors that correspond to the k largest eigenvalues to construct a dxk-dimensional transformation matrix, W; the eigenvectors are the columns of this matrix
7. Project the examples onto the new feature subspace using the transformation matrix, W

LDA takes class label information into account, represented by the `mean vectors` from step 2

#### Computing Scatter Matrices

```python
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
```

We assume that the class labels in the training dataset are uniformly distributed but if we print them they are not

```python
print("Class label distribution:", np.bincount(y_train)[1:])
```

We need to scale the individual scatter matrices S_i, before we sum them up as the scatter matrix S_w

```python
d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
class_scatter = np.cov(X_train_std[y_train == label].T)
S_W += class_scatter
print("Scaled within-class scatter matrix: " f"{S_W.shape[0]}x{S_W.shape[1]}")
```

Compute the Between-class scatter matrix

```python
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)

d = 13 # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
n = X_train_std[y_train == i + 1, :].shape[0]
mean_vec = mean_vec.reshape(d, 1) # make column vector
S_B += n \* (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print("Between-class scatter matrix: " f"{S_B.shape[0]}x{S_B.shape[1]}")
```

Selecting linear discriminants for the new feature subspace

```python
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

eigen_pairs = [
(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]

eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

print("Eigenvalues in descending order:\n")
for eigen_val in eigen_pairs:
print(eigen_val[0])

```

> NOTE: in LDA the number of linear discriminants is **AT MOST** c - 1, where c is the `number of class labels`, since the in-between scatter matrix, $S_B$ is the sum of c matrices with rank one or less.

To view how much of the overall variance is explained by the linear discriminants we can plot them by decreasing eigenvalues:

```python
tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14), discr, align='center', label='Individual discriminability')
plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative discriminability')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

Stack the two most discriminative eigenvector columns to create the transformation matrix, W:

```python
w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))

print('Matrix W: \n', w)
```

#### Projecting examples onto the new feature subspace

Using the transformation matrix, W, we can now transform the training dataset by matrix multiplication: $X' = XW$

```python
X_train_lda = X_train_std.dot(w)
colors=['r', 'b', 'g']
markers=['o', 's', '^']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l  , 0], X_train_lda[y_train==l,1]*(-1),
    c=c, label=f'Class {l}', marker=m)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

### LDA in Scikit-Learn

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

# Use a LogisticRegression classifer with the LDA feature space
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
```

## Nonlinear Dimensionality Reduction and visualization

### t-distributed Stochastic Neighbor Embedding (t-SNE)

Used to visualize high-dimensional datasets into two or three dimensions.

#### Why consider non-linear dimensionality reduction?

Many machine learning algorithms make assumptions about the linear separability of the input data.

Development and application of nonlinear dimensionality reduction techniques is also often referred to as `manifold learning` where a `manifold` refers to a lower dimensional topological space embedded in a high-dimensional space.

Manifold algorithms have to capture the complicated structure of the data in order to project it onto a lower-dimensional space where the relationship between data pionts is preserved.

> NOTE: These techniques are notoriously hard to use and with non-ideal hyperparameter choices, they may cause more harm than good. Unless we project the data into 2D or 3D (often not sufficient to capture more complicated relationships), it is hard or impossible to assess the quality of the results.

t-SNE learns to embed data points into a lower-dimensional space such that the pairwise distances in the original space are preserved.

> NOTE: t-SNE is primiarily used for visualization purposes and requires the **WHOLE** dataset for the projection--since it projects the points directly (no projection matrix), we cannot apply t-SNE to new data points.

```python
# Example of t-SNE applied to a 64 dimension dataset
import matplotlib.patheffects as PathEffects
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

digits = load_digits()
fig, ax = plt.subplots(1,4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show()

print(digits.data.shape)

# Assign the features (pixels) to a new variable X_digits and the labels to another new variable y_digits
y_digits = digits.target
X_digits = digits.data

tsne=TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)

# Visualize the 2D t-SNE embeddings
def plot_projection(x, colors):
    f = plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0],
        x.colors ==i, 1])

    for i in range(10):
        xtext, ytext = np.median(x[colors==i, ;], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
        PathEffects.Stroke(linewidth=5, foreground="w"),
        PathEffects.Normal()])

plot_projection(X_digits_tsne, y_digits)
plt.show()
```
