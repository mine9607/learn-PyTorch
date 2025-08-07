import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

sys.path.append(os.path.abspath("../2_Simple-Classification"))
from utils import plot_decision_regions

np.random.seed(1)

# Generate Xor Data
# np.random.randn returns 200 examples of 2 dimensions (i.e. [[1, 2], [3,7]) from a normal distribution
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)

# Prepare SVM models with different gamma values
svm1 = SVC(kernel="rbf", random_state=1, gamma=0.10, C=10.0)
svm2 = SVC(kernel="rbf", random_state=1, gamma=0.20, C=10.0)

# Fit Models
svm1.fit(X_xor, y_xor)
svm2.fit(X_xor, y_xor)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Plot raw XOR data
ax = axes[0]
ax.scatter(
    X_xor[y_xor == 1, 0],
    X_xor[y_xor == 1, 1],
    c="royalblue",
    marker="s",
    label="Class 1",
)
ax.scatter(
    X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], c="tomato", marker="o", label="Class 0"
)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_title("XOR Data")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend(loc="best")

# 2. Plot decision region with gamma=0.01
plot_decision_regions(X_xor, y_xor, classifier=svm1, ax=axes[1])
axes[1].set_title("SVM with RBF Kernel (gamma=0.01)")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend(loc="upper left")

# 3. Plot decision region with gamma=0.2
plot_decision_regions(X_xor, y_xor, classifier=svm2, ax=axes[2])
axes[2].set_title("SVM with RBF Kernel (gamma=0.2)")
axes[2].set_xlabel("Feature 1")
axes[2].set_ylabel("Feature 2")
axes[2].legend(loc="upper left")

plt.tight_layout()
plt.show()
