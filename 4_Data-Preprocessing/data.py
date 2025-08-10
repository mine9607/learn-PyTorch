from io import StringIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

csv_data = """A, B, C, D
        1.0, 2.0, 3.0, 4.0
        5.0, 6.0,, 8.0
        10.0, 11.0, 12.0,"""

df = pd.read_csv(StringIO(csv_data))

# Print Dataframe
print(df, "\n")

# Print sum of null values per column
print(df.isnull().sum(), "\n")

# Print the np array from the Dataframe (used for input to sklearn functions/methods)
print("Values", df.values, "\n")

# Drop examples (rows) or features (columns)
df_drop_row = df.dropna(axis=0)
df_drop_col = df.dropna(axis=1)

print("Original DF\n", df, "\n")
print("Drop Row DF\n", df_drop_row, "\n")
print("Drop Col DF\n", df_drop_col, "\n")

# Get info about the data frame
print(df.info())

# ---------------------------------------------------------------------------------

# MEAN IMPUTE OF MISSING DATA

# Sklearn impute method
imr = SimpleImputer(missing_values=np.nan, strategy="mean")
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print("Sklearn Imputer Class")
print(imputed_data)

# pandas impute method (fillna)
print("Pandas fillna Method")
print(df.fillna(df.mean()))

# -----------------------------------------------------------------------------------

# ENCODING CATEGORICAL DATA (ORDINAL VS NOMINAL)

df2 = pd.DataFrame(
    [
        ["green", "M", 10.1, "class2"],
        ["red", "L", 13.5, "class1"],
        ["blue", "XL", 15.3, "class2"],
    ]
)

df2.columns = ["color", "size", "price", "classlabel"]
print("\n", df2)

# Manual Feature Enconding ------------------------------------------

# Encode Feature Data - # Define a mapping
size_mapping = {"XL": 3, "L": 2, "M": 1}

df2["size"] = df2["size"].map(size_mapping)
print("\n", df2)

# Decode feature data
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print("\n", df2["size"].map(inv_size_mapping))

df2["size"] = df2["size"].map(inv_size_mapping)
print("\n", df2)

# Manual Mapping of Class Labels --------------------------------------
class_mapping = {label: idx for idx, label in enumerate(np.unique(df2["classlabel"]))}
print("\n", class_mapping, "\n")

# Encode class labels
df2["classlabel"] = df2["classlabel"].map(class_mapping)
print("Encoded Labels\n", df2, "\n")

# Decode label data
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df2["classlabel"] = df2["classlabel"].map(inv_class_mapping)
print("Decoded\n", df2, "\n")

# ENCODING CLASS LABELS WITH SKLEARN ----------------------------------
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()

# Returns a list of encoded labels (e.g. [1, 0, 1])
y = class_le.fit_transform(df2["classlabel"].values)
print(y)

print(class_le.inverse_transform(y))

# ENCODING NOMINAL DATA

# One-hot Encoding
print("\nONE HOT ENCODING\n")
X = df2[["color", "size", "price"]].values
color_ohe = OneHotEncoder(drop="first", categories="auto")
result = color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
print(result)

# To selectively manipulate Categorical Columns Use COlumnTransformer
print("\nColumn Transformer\n")
c_transf = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse=False), [0]),
    ],
    remainder="drop",
)

results = c_transf.fit_transform(X).astype(float)
print(results)

# GET DUMMIES

df2 = pd.get_dummies(df2[["price", "color", "size"]], drop_first=True)
print(df2)

# --------------------------------------------------------------------------

# TRAIN TEST SPLIT

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/" "ml/machine-learning-databases/" "wine/wine.data",
    header=None,
)

df_wine.info()
print(df_wine.head())

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

print("Class labels", np.unique(df_wine["Class Label"]))
print(df_wine.head())

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# --------------------------------------------------------------------------------------
# FEATURE SCALING

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

# FEATURE SELECTION

# KNN

# DECISION TREE / RANDOM FOREST
