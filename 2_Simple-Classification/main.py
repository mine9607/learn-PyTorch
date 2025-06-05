import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron import Perceptron
from utils import plot_decision_regions

# Download the iris dataset
DATA_PATH = "iris.data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

if not os.path.exists(DATA_PATH):
    import urllib.request

    print("Downloading dataset...")
    urllib.request.urlretrieve(URL, DATA_PATH)


df: pd.DataFrame = pd.read_csv(DATA_PATH, header=None, encoding="utf-8")  # type: ignore

# Display the last 5 items from the dataframe
print(df.tail())

# Select Setosa and Versicolor

y = df.iloc[0:100, 4].values

y = np.where(y == "Iris-setosa", 0, 1)

# extract sepal length and petal length

X = df.iloc[0:100, [0, 2]].values

# plot data

plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="s", label="Versicolor")

plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc="upper left")
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of Updates")
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc="upper left")
plt.autoscale()
plt.show()
