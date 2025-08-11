import os
import sys

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
