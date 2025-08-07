from io import StringIO

import numpy as np
import pandas as pd

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

df_drop_row = df.dropna(axis=0)
df_drop_col = df.dropna(axis=1)

print("Original DF\n", df, "\n")
print("Drop Row DF\n", df_drop_row, "\n")
print("Drop Col DF\n", df_drop_col, "\n")
# print(df.info())

# Mean Impute
from sklearn.impute import SimpleImputer

imr = SimpleImputer(missing_values=np.nan, strategy="mean")
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
# sklearn Imputer
print("Sklearn Imputer Class")
print(imputed_data)

# pandas impute method (fillna)
print("Pandas fillna Method")
print(df.fillna(df.mean()))

### Categorical Encoding

df2 = pd.DataFrame(
    [
        ["green", "M", 10.1, "class2"],
        ["red", "L", 13.5, "class1"],
        ["blue", "XL", 15.3, "class2"],
    ]
)

df2.columns = ["color", "size", "price", "classlabel"]

print("\n", df2)

# Define a mapping for converting ordinal sizes to numeric data
size_mapping = {"XL": 3, "L": 2, "M": 1}

df2["size"] = df2["size"].map(size_mapping)
print("\n", df2)

# Convert numeric data back to ordinal sizes
inv_size_mapping = {v: k for k, v in size_mapping.items()}

print("\n", df2["size"].map(inv_size_mapping))


df2["size"] = df2["size"].map(inv_size_mapping)

print("\n", df2)

# Use the index of the array of unique values as the class label to create the map
class_mapping = {label: idx for idx, label in enumerate(np.unique(df2["classlabel"]))}
print("\n", class_mapping, "\n")

# Encode class labels
df2["classlabel"] = df2["classlabel"].map(class_mapping)
print("Encoded Labels\n", df2, "\n")

# Decode label data
inv_class_mapping = {v: k for k, v in class_mapping.items()}

df2["classlabel"] = df2["classlabel"].map(inv_class_mapping)
print("Decoded\n", df2, "\n")

from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
# Returns a list of encoded labels (e.g. [1, 0, 1])
y = class_le.fit_transform(df2["classlabel"].values)
print(y)

print(class_le.inverse_transform(y))
