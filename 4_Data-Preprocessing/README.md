# Chapter 4

## Building Good Training Datasets - Data Preprocessing

- Remove and impute missing values from the dataset
- Transform categorical data into shape for ML algorithms
- Feature selection

Best Practice Steps:

1. Remove or Impute missing data (features = cols or examples = rows)
2. Encode Categorical Features or Labels
   - For "Nominal Features" use `one-hot encoding` (i.e. create dummy features for each unique value in the column)
   - For "Ordinal Features" create a mapping
   - For "Class Labels" assign the idx of the array np.unique() method to each value via label map or sklearn LabelEncoder

### Dealing with missing data

#### View CSV Data in Dataframe

```python

# read csv data into a dataframe
df = pd.read_csv()

# print for viewing
print(df)
```

#### Check columns for missing values

```python
# Sum false values per column
print(df.is_null().sum())
```

## Drop examples (rows) or features (columns) with missing data

```python
# Drop row
df.dropna(axis=0)

# Drop col
df.dropna(axis=1)
```

## Interpolating Missing Values

### Mean Imputation

Replaces the missing values with the mean value of the entire feature column

#### Sklearn Imputer Class

```python
from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)
```

#### Pandas 'fillna' method

```python
print(df.fillna(df.mean()))
```

## Categorical Data Encoding

Ordinal features = categorical values that can be `sorted` and `ordered` (e.g. t-shirt size)
Nominal features = categorical values that **DON'T** imply any order (e.g. t-shirt color)

### Encoding Features (Ordinal)

#### Manual Feature Mapping

Create a map to apply to the column with ordinal data:

```python
# Create map of ordinal values to numeric data
size_mapping = {"XL":3, "L":2, "M":1}
# Apply map to ordinal feature column 'size'
df['size'] = df['size'].map(size_mapping)
```

### Encoding Class Labels (Nominal)

#### Manual Label Encoding

> Note: Class labels are NOT ordinal and it doesn't matter which integer number we assign to a label

```python
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}

print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
```

#### Sklearn LabelEncoder

```python
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

# Encode class labels -> returns a list (e.g. [1,0,1])
y = class_le.fit_transform(df['classlabel'].values)
print(y)

# decode class labels
y = class_le.inverse_transform(y)
print(y)
```

### Encoding Features (Nominal)

#### One-hot Encoding

Use one-hot encoding to ensure classification model doesn't associate an order with the values

```python
from sklearn.preprocessing import OneHotEncoder
# Remove the 'class labels column from the dataframe and create an array of the values
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
# Apply the ecoder to only the color column (X[:,0].reshape(-1,1))
color_values = color_ohe.fit_transform(X[:, 0].reshape(-1,1)).toarray()
print(color_values)
```

#### Column Transformer

```python
from sklearn.compose import ColumnTransformer

X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse=False), [0]),
    ],
    remainder="drop",
)

results = c_transf.fit_transform(X).values
print(results)
```

#### Get Dummies

```python
pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)
```

## Partitioning Data: Train Test Split

```python
from sklearn.model_selection import train_test_split

```
