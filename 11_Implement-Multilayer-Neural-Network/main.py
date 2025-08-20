import matplotlib.pyplot as plt
from mlp import NeuralNetMLP
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.values
y = y.astype(int).values

print(X.shape)
print(y.shape)

# Normalize the pixel values in the MNIST to the range -1 to 1 (originally 0 to 255)
X = ((X / 255.0) - 0.5) * 2

# Plot an example of each digit
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

ax = ax.ravel()

for i in range(10):
    img = X[y == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap="Greys")

ax[0].set_xticks([])
ax[0].set_yticks([])

plt.tight_layout()
plt.show()

# Plot multiple examples of the same digit
fig, ax = plt.subplots(5, 5, sharex=True, sharey=True)

ax = ax.ravel()

for i in range(25):
    img = X[y == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap="Greys")

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Split data into train, test
# First split into training subset (60,000) and test set (10,000)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y
)

# Second split training subset into training (55,000) and validation (5000) subsets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp
)

model = NeuralNetMLP(num_features=28 * 28, num_hidden=50, num_classes=10)
