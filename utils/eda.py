import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer

"""
This is a simple python file to practice Exploratory Data Analysis (EDA) by building plots.

Included Plots:

Method 1
    A) Before Modeling - EDA Plots
        1. Histograms (KDE Plots)
            - seaborn.histplot or sns.kdeplot
        2. Bar plots--for categorical features
            - sns.countplot
        3. Correlation heatmap--for numeric features
            - sns.heatmap(df.corr())
        4. Boxplots or Violin plots vs target
            - sns.boxplot(x=target, y=feature)
        5. Pair Plot (scatter matrix)
            - sns.pairplot
    B) During/After Modeling - Model & Training Plots
        6. Learning Curves
            - sklearn.model_selection.learning_curve
        7. Validation Curves
            - sklearn.model_selection.validation_curve
        8. ROC curve & Precision-Recall curve
            - sklearn.metrics.roc_curve, precision_recall_curve
        9. Confusion Matrix heatmap
            - sns.heatmap(confusion_matrix(...))
        10. Feature importance SHAP summary plot
            - model.feature_importances_ or shap.summary_plot

Method 2: Unsupervised Plots vs Supervised Plots

    Supervised:
        Histograms/KDE plots
        Correlation heatmap
        Pair Plot
        2D projection (PCA, t-SNE, UMAP)
        Cluster visualizations
        Feature Variance plot

    NLP / High-Dimensional Data:
        Token/term frequency plots
        Document length distribution
        Word clouds
        Dimensionality reduction scatter
        Heatmap of cosine similarity

    Advanced Models/Deep Learning:
        Training vs Valication loss Curves
        Metric Curves per Epoch
        Embedding Visualizations
        Attention heatmaps

"""

# Note this returns the following:
"""
data -> a pandas DataFrame of the features only
target -> a pandas Series of the target labels
frame -> a complete pandas DataFrame of the Features and Labels
feature_names -> a list of feature names (column titles)
target_names -> a list of target names ('malignant' vs 'benign') 
DESCR -> dataset description
"""

# Download the Dataset
data = load_breast_cancer(as_frame=True)

# Get Data Description
# print(data.DESCR)

# Get Full data frame
df = data.frame

# Inspect the dataframe
print(df.info())

# Check for null values
# print(df.isnull().sum())

# Get class distribution
class_count = df["target"].value_counts()
class0_count = class_count[0]
class1_count = class_count[1]

print("Class 0 Count: ", class0_count, "\nClass 1 Count: ", class1_count)

# Split into X and y values
X = data.data.values
y = data.target.values


# Plot histogram of one feature
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.hist(df["mean radius"])
ax.set_title("Mean Radius", fontsize=14, pad=10)
ax.set_xlabel("Bin")
ax.set_ylabel("Count")
plt.show()

# Plot one feature using Seaborn
sns.histplot(df["mean radius"], kde=True)
plt.show()
# sns.kdeplot(df["mean radius"])
# plt.show()

# Loop over each feature and plot ALL FEATURES
# fig, axes = plt.subplots(6, 5, figsize=(14, 8), constrained_layout=True)
# axes = axes.ravel()
# features = data.feature_names
# for ax, f in zip(axes, features):
#     ax.hist(df[f])
#     ax.set_title(f)
# plt.show()

# Plot all featurs as Seaborn Hists
fig, axes = plt.subplots(6, 5, figsize=(14, 8), constrained_layout=True)
axes = axes.ravel()
features = data.feature_names
for ax, f in zip(axes, features):
    sns.histplot(df[f], ax=ax, kde=True)
    ax.set_title(f)
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.show()


# Plot frequency for categorical features (none-present)
"""
use value_count() for each categorical column and plot with sns.countplot()

automate so it loops through all object-type columns (use column dtype)

"""

# Correlation Heatmap
corr_matrix = df.drop(columns="target").corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# Compute correlations with target
target_corr = df.corr()["target"].sort_values(ascending=False)

# Reorder DataFrame columns by this correlation
sorted_features = target_corr.index
corr_matrix_sorted = df[sorted_features].corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix_sorted, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Features Sorted by Correlation with Target")
plt.show()

# Plot boxplot
sns.boxplot(x=df["target"], y=df["mean radius"])
plt.show()

sns.violinplot(x=df["target"], y=df["mean radius"])
plt.show()
