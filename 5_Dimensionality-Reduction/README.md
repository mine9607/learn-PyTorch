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
