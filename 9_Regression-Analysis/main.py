import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.plotting import heatmap, scatterplotmatrix
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.abspath("../2_Simple-Classification"))
from linear_reg import LinearRegressionGD

columns = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
]

df = pd.read_csv(
    "http://jse.amstat.org/v19n3/decock/AmesHousing.txt", sep="\t", usecols=columns
)

print(df.head())
print(df.shape)

# Convert the "Central Air" feature column to integers
df["Central Air"] = df["Central Air"].map({"N": 0, "Y": 1})

# Check for missing values
print(df.info())
print(df.isnull().sum())

# Drop example missing the 'Total Bsmt SF' data
df = df.dropna(axis=0)
print(df.info())
print(df.isnull().sum())

# Visualize the Feature Correlations
scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)

plt.tight_layout()
plt.show()

# Visualize the Data - Correlation Matrix
print("Correlation Matrix:\n")
# note np.corrcoef expects each feature to be a row not column so we first need to transpose the data
cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.show()

# Train Linear Regression Class model on Gr Liv Area to predict sale price
X = df[["Gr Liv Area"]].values
y = df["SalePrice"].values

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)
# print("y-shape:", y.shape)
# print("y-shape_new-axis:", y[:, np.newaxis].shape)
# print("y-shape_new-axis_flat:", y[:, np.newaxis].flatten().shape)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD(eta=0.1)

lr.fit(X_std, y_std)

# Plot the error per epoch
# ## remember we defined the logistic regression class to calculate MSE as the loss function
plt.plot(range(1, lr.n_iter + 1), lr.losses_)
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.show()


# Plot the Sale Price vs Living Area with prediction
def lin_regplt(X, y, model):
    plt.scatter(X, y, c="steelblue", edgecolor="white", s=70)
    plt.plot(X, model.predict(X), color="black", lw=2)


lin_regplt(X_std, y_std, lr)
plt.xlabel("Living area above ground (standardized)")
plt.ylabel("Sale Price (standardized)")

plt.show()


# Convert standardized features back to original
feature_std = sc_x.transform(np.array([[2500]]))
target_std = lr.predict(feature_std)
target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
print(f"Sales price: ${target_reverted.flatten()[0]:.2f}")

# Check model parameters of standardized inputs
print(f"Slope: {lr.w_[0]:.3f}")
print(f"Intercept: {lr.b_[0]:.3f}")


# Estimate the coefficient of a regression model - Scikit-learn
slr = LinearRegression()

slr.fit(X, y)
y_pred = slr.predict(X)
print(f"Slope: {slr.coef_[0]:.3f}")
print(f"Intercept: {slr.intercept_:.3f}")

lin_regplt(X, y, slr)
plt.xlabel("Living area above ground (ft2)")
plt.ylabel("Sale Price (USD)")
plt.tight_layout()
plt.show()

# RANSAC - fit the model to a random subset of the data (in-liers)
ransac = RANSACRegressor(
    LinearRegression(),
    max_trials=100,  # default
    min_samples=0.95,
    residual_threshold=None,  # defuault
    random_state=123,
)

ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(
    X[inlier_mask],
    y[inlier_mask],
    c="steelblue",
    edgecolor="white",
    marker="o",
    label="Inliers",
)
plt.scatter(
    X[outlier_mask],
    y[outlier_mask],
    c="limegreen",
    edgecolor="white",
    marker="s",
    label="Outliers",
)
plt.plot(line_X, line_y_ransac, color="black", lw=2)
plt.xlabel("Living area above ground (ft2)")
plt.ylabel("Sale price (USD)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

print(f"Slope: {ransac.estimator_.coef_[0]:.3f}")
print(f"Intercept: {ransac.estimator_.intercept_:.3f}")


def median_absolute_deviation(data):
    return np.median(np.abs(data - np.median(data)))


print(median_absolute_deviation(y))

# Evaluate the performance of Linear Regression Models

target = "SalePrice"
features = df.columns[df.columns != target]
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# Make a residual plot

x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

ax1.scatter(
    y_test_pred,
    y_test_pred - y_test,
    c="limegreen",
    marker="s",
    edgecolor="white",
    label="Test Data",
)

ax2.scatter(
    y_train_pred,
    y_train_pred - y_train,
    c="steelblue",
    marker="o",
    edgecolor="white",
    label="Train Data",
)
ax1.set_ylabel("Residuals")

for ax in (ax1, ax2):
    ax.set_xlabel("Predicted values")
    ax.legend(loc="upper left")
    ax.hlines(y=0, xmin=x_min - 100, xmax=x_max + 100, color="black", lw=2)

plt.tight_layout()
plt.show()

# Evaluate MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"MSE train: {mse_train:.3f}")
print(f"MSE test: {mse_test:.3f}")

# Evaluate MAE
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f"MAE train: {mae_train:.3f}")
print(f"MAE test: {mae_test:.3f}")

# Evaluate R2 Score
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"R^2 train: {r2_train:.3f}")
print(f"R^2 test: {r2_test:.3f}")

# Using Regularized Methods for Regression

# - Ridge Regression - L2 penalized model
# - Least Absolute Shrinkage and Selection Operator (LASSO) - creates sparse vectors L1 regularization
# - Elastic net - has both L1 and L2 regularization

# Adding Polynomial terms using scikit-learn

X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[
    :, np.newaxis
]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])
lr = LinearRegression()
pr = LinearRegression()

lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

plt.scatter(X, y, label="Training points")
plt.plot(X_fit, y_lin_fit, label="Linear fit", linestyle="--")
plt.plot(X_fit, y_quad_fit, label="Quadratic fit", linestyle="--")
plt.xlabel("Explanatory variable")
plt.ylabel("Predicted or known target values")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# Compute the MSE and R2 metrics
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
mse_lin = mean_squared_error(y, y_lin_pred)
mse_quad = mean_squared_error(y, y_quad_pred)
print(f"Training MSE Linear: {mse_lin:.3f}, quadratic: {mse_quad:.3f}")

r2_lin = r2_score(y, y_lin_pred)
r2_quad = r2_score(y, y_quad_pred)
print(f"Training R^2 Linear: {r2_lin:.3f}, quadratic: {r2_quad:.3f}")

# Apply Quadratic Fits to Housing Data

# Remove outliers
X = df[["Gr Liv Area"]].values
y = df["SalePrice"].values
X = X[(df["Gr Liv Area"] < 4000)]
y = y[(df["Gr Liv Area"] < 4000)]

regr = LinearRegression()

# create quadratic anc cubic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Fit to features
X_fit = np.arange(X.min() - 1, X.max() + 2, 1)[:, np.newaxis]

regr = regr.fit(X, y)

y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

y_cubic_fit = regr.predict(quadratic.fit_transform(X_fit))
regr = regr.fit(X_cubic, y)
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# Plot results
plt.scatter(X, y, label="Training points", color="lightgray")
plt.plot(
    X_fit,
    y_lin_fit,
    label=f"Linear (d=1), $R^2$={linear_r2:.2f}",
    color="blue",
    lw=2,
    linestyle=":",
)
plt.plot(
    X_fit,
    y_quad_fit,
    label=f"Quadratic (d=2), $R^2$={quadratic_r2:.2f}",
    color="red",
    lw=2,
    linestyle="-",
)
plt.plot(
    X_fit,
    y_cubic_fit,
    label=f"Cubic (d=3), $R^2$={cubic_r2:.2f}",
    color="green",
    lw=2,
    linestyle="--",
)

plt.xlabel("Living area (ft2)")
plt.ylabel("Sale price (USD)")
plt.legend(loc="upper left")
plt.show()

X = df[["Overall Qual"]].values
y = df["SalePrice"].values


regr = LinearRegression()

# create quadratic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# fit features
X_fit = np.arange(X.min() - 1, X.max() + 2, 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))


# plot results
plt.scatter(X, y, label="Training points", color="lightgray")

plt.plot(
    X_fit,
    y_lin_fit,
    label=f"Linear (d=1), $R^2$={linear_r2:.2f}",
    color="blue",
    lw=2,
    linestyle=":",
)

plt.plot(
    X_fit,
    y_quad_fit,
    label=f"Quadratic (d=2), $R^2$={quadratic_r2:.2f}",
    color="red",
    lw=2,
    linestyle="-",
)

plt.plot(
    X_fit,
    y_cubic_fit,
    label=f"Cubic (d=3), $R^2$={cubic_r2:.2f}",
    color="green",
    lw=2,
    linestyle="--",
)


plt.xlabel("Overall quality of the house")
plt.ylabel("Sale price in U.S. dollars")
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()

# Decision Tree Regression

X = df[["Gr Liv Area"]].values
y = df["SalePrice"].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()
lin_regplt(X[sort_idx], y[sort_idx], tree)
plt.xlabel("Living area (ft2)")
plt.ylabel("Sale Price (USD)")
plt.tight_layout()
plt.show()
