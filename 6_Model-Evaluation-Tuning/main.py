import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from numpy import interp
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    header=None,
)

print(df.head())
# df.info()

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
# All columns are indexed not text based (i.e. 0, 1, 2 ... 31)

# Create column labels for values in column 1
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

print(le.transform(["M", "B"]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

# Combine transformers and estimators in a pipeline

# We need to standardize the feature set and lets assume we want to use PCA for dimensionality reduction

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())

pipe_lr.fit(X_train, y_train)

y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}")


# Using sklearn Stratified K-fold

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(
        f"Fold: {k+1:02d}, "
        f"Class distr.: {np.bincount(y_train[train])},"
        f"Acc.: {score:.3f}"
    )

mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f"\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")

# Use sklearn cross_val_score
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print(f"CV accuracy scores: {scores}")
print(f"CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")


# Learning curve from scikit-learn
"""
Here we are splitting the TRAINING dataset into 10, 20, 30, 40, 50, 60, 70, 80, 90, 100% and for each performing a k-fold cross-validation, k=10, (we are varying the amount of training data--NOT hyperparameters or model) 
"""

piple_lr = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2", max_iter=10000)
)

# note: here "test_scores" are the results of the k-folds evaluations (i.e. validation data INSIDE training dataset) NOT the holdout "unseen" data
train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=10,
    n_jobs=1,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(
    train_sizes,
    train_mean,
    color="blue",
    marker="o",
    markersize=5,
    label="Training accuracy",
)
plt.fill_between(
    train_sizes,
    train_mean + train_std,
    train_mean - train_std,
    alpha=0.15,
    color="blue",
)
plt.plot(
    train_sizes,
    test_mean,
    color="green",
    linestyle="--",
    marker="s",
    markersize=5,
    label="Validation accuracy",
)
plt.fill_between(
    train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green"
)
plt.grid()
plt.title("Learning Curve - Training Set Size")
plt.xlabel("Number of training examples")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.ylim([0.8, 1.03])
plt.show()

# ## Validation Curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name="logisticregression__C",
    param_range=param_range,
    cv=10,
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(
    param_range,
    train_mean,
    color="blue",
    marker="o",
    markersize=5,
    label="Training accuracy",
)
plt.fill_between(
    param_range,
    train_mean + train_std,
    train_mean - train_std,
    alpha=0.15,
    color="blue",
)
plt.plot(
    param_range,
    test_mean,
    color="green",
    linestyle="--",
    marker="s",
    markersize=5,
    label="Validation accuracy",
)
plt.fill_between(
    param_range,
    test_mean + test_std,
    test_mean - test_std,
    alpha=0.15,
    color="green",
)
plt.grid()
plt.xscale("log")
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.ylim([0.8, 1.0])
plt.title("Validation Curve - Inverse Regularization (C)")
plt.show()

# Grid Search Hyperparameter Optimization
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Define the parameters of the chosen estimator (in this case kernel and gamma for SVC)
param_grid = [
    {"svc__C": param_range, "svc__kernel": ["linear"]},
    {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]},
]

gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring="accuracy",
    cv=10,
    refit=True,
    n_jobs=-1,
)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")


# Randomized Search for tuning an SVC
param_range = scipy.stats.loguniform(0.0001, 1000.0)

np.random.seed(1)
print(param_range.rvs(10), "\n")

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_grid = [
    {"svc__C": param_range, "svc__kernel": ["linear"]},
    {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]},
]

rs = RandomizedSearchCV(
    estimator=pipe_svc,
    param_distributions=param_grid,
    scoring="accuracy",
    refit=True,
    n_iter=20,
    cv=10,
    random_state=1,
    n_jobs=-1,
)

rs = rs.fit(X_train, y_train)
print("Randomized Search Best Accuracy:", rs.best_score_)
print("Randomized Search Best Estimator:", rs.best_params_)

# Successive Halving
hs = HalvingRandomSearchCV(
    pipe_svc,
    param_distributions=param_grid,
    n_candidates="exhaust",
    resource="n_samples",
    factor=1.5,
    random_state=1,
    n_jobs=-1,
)

hs = hs.fit(X_train, y_train)
print(hs.best_score_)
print(hs.best_params_)
clf = hs.best_estimator_
print(f"Test accuracy: {hs.score(X_test, y_test):.3f}")

# Nested Cross-Validation

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
    {"svc__C": param_range, "svc__kernel": ["linear"]},
    {"svc__C": param_range, "svc__gamma": param_range, "svc__kernel": ["rbf"]},
]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring="accuracy", cv=2)

scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)
print(f"CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

# NOTE: the returned average cross-validation accuracy gives us a good estimate of what to expect if we tune the hyperparameters of a model and use it on unseen data

# Simple example comparing SVM model to a simple decision tree classifier

gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[{"max_depth": [1, 2, 3, 4, 5, 6, 7, None]}],
    scoring="accuracy",
    cv=2,
)

scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)

print(f"CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

# Confusion Matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")

ax.xaxis.set_ticks_position("bottom")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()


pre_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f"Precision: {pre_val:.3f}")

rec_val = recall_score(y_true=y_test, y_pred=y_pred)
print(f"Recall: {rec_val:.3f}")

f1_val = f1_score(y_true=y_test, y_pred=y_pred)
print(f"F1: {f1_val:.3f}")

mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
print(f"MCC: {mcc_val:.3f}")

# RECEIVER OPERATING CHARACTERISTIC

pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(penalty="l2", random_state=1, solver="lbfgs", C=100.0),
)

X_train2 = X_train[:, [4, 14]]
cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC fold {i+1} (area = {roc_auc:.2f})")

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color=(0.6, 0.6, 0.6),
    label="Random guessing (area=0.5)",
)

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, "k--", label=f"Mean ROC (area = {mean_auc:.2f})", lw=2)
plt.plot(
    [0, 0, 1],
    [0, 1, 1],
    linestyle=":",
    color="black",
    label="Perfect performance (area=1.0)",
)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc="lower right")
plt.show()

# Imbalanced Classes

X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

y_pred = np.zeros(y_imb.shape[0])
print(np.mean(y_pred == y_imb) * 100)

print("Number of class 1 examples before:", X_imb[y_imb == 1].shape[0])

# Resample minority class with replacement
X_upsampled, y_upsampled = resample(
    X_imb[y_imb == 1],
    y_imb[y_imb == 1],
    replace=True,
    n_samples=X_imb[y_imb == 0].shape[0],
    random_state=123,
)

print("Number of class 1 examples after:", X_upsampled.shape[0])

# Stack original class 0 samples with the upsampled class 1 subset to obtain a balanced dataset

X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

y_pred = np.zeros(y_bal.shape[0])
print(np.mean(y_pred == y_bal) * 100)
