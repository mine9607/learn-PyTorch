import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adaline import Adaline
from perceptron import Perceptron
from utils import plot_decision_regions

# Download the iris dataset
DATA_PATH = "iris.data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Check if the path exists and if not download the data to to the path
if not os.path.exists(DATA_PATH):
    import urllib.request

    print("Downloading dataset...")
    urllib.request.urlretrieve(URL, DATA_PATH)


df: pd.DataFrame = pd.read_csv(DATA_PATH, header=None, encoding="utf-8")  # type: ignore

# Display the last 5 items from the dataframe
print(df.tail())

# Select Setosa and Versicolor
# Take column 4 of the first 100 rows of the dataframe
y = df.iloc[0:100, 4].values

# Use one-hot encoding to create a labeled data set of 0 == "Iris-setosa" and 1 == "Not Iris-Setosa"
y = np.where(y == "Iris-setosa", 0, 1)

# extract sepal length and petal length for the first 100 rows, select columns by index position (first and third) and convert to a numpy array (.values)
X = df.iloc[0:100, [0, 2]].values

# plot data

plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="s", label="Versicolor")

plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc="upper left")
plt.show()

ppn = Perceptron(eta=0.1, n_iter=20)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of Updates")
plt.show()

aln = Adaline(eta=0.001, n_iter=100)
aln.fit(X, y)
plt.plot(range(1, len(aln.losses_) + 1), aln.losses_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of Updates")
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = Adaline(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker="o")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(Mean squared error)")
ax[0].set_title("Adaline - Learning Rate 0.1")

ada2 = Adaline(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker="o")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Mean squared error")
ax[1].set_title("Adaline - Learning Rate 0.0001")
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.title("Perceptron Classifier - LR=0.1")
plt.legend(loc="upper left")
plt.autoscale()
plt.show()

# Standardizing the dataset
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()  # standardize sepal length
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()  # standardize petal length

ada_std = Adaline(n_iter=20, eta=0.5)
ada_std.fit(X_std, y)

plot_decision_regions(X, y, classifier=aln)
plt.xlabel("Sepal lenngth [cm]")
plt.ylabel("Petal length [cm]")
plt.title("Adaline Classifier - LR=0.001")
plt.legend(loc="upper left")
plt.autoscale()
plt.show()

plot_decision_regions(X_std, y, classifier=ada_std)
plt.title("Adaline - Gradient Descent - Standardized")
plt.xlabel("Sepal length [standardized]")
plt.ylabel("Petal length [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada_std.losses_) + 1), ada_std.losses_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Mean squared error")
plt.tight_layout()
plt.show()
