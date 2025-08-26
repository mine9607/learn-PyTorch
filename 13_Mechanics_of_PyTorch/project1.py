# Predicting Fuel Efficiency of a Car (Regression)

"""
This project will follow the following workflow:

1 - Data Preprocessing
2 - Feature Engineering
3 - Training
4 - Prediction (Inference)
5 - Evaluation

"""
import numpy as np
import pandas as pd
import torch
from numpy.core import numerictypes
from sklearn.model_selection import train_test_split

# Working with Feature Columns

## Get data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

df = pd.read_csv(
    url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True
)

print(df.info())
print(df.isnull().sum())

## Drop the NA rows
df = df.dropna()
# reset the index after dropping
df = df.reset_index(drop=True)

## train/test split:
df_train, df_test = train_test_split(df, train_size=0.8, random_state=1)

train_stats = df_train.describe().transpose()

numeric_column_names = [
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
]

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, "mean"]
    std = train_stats.loc[col_name, "std"]
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean) / std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std

print(df_train_norm.tail())

## re-classify 'model year' to 0, 1, 2, 3 {year < 73 = 0, 73 <= year <= 76 = 1...etc}
boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_norm["Model Year"].values)
df_train_norm["Model Year Bucketed"] = torch.bucketize(v, boundaries, right=True)

v = torch.tensor(df_test_norm["Model Year"].values)
df_test_norm["Model Year Bucketed"] = torch.bucketize(v, boundaries, right=True)

numeric_column_names.append("Model Year Bucketed")
