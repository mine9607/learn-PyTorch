import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# returns values from [-7 to 6.9) non-inclusive
z = np.arange(-7, 7, 0.1)

# returns an inclusive range [-7 to 7] with equal interval spacing of 0.1
y = np.linspace(-7, 7, 141)

print(z)
print(y)

sigma_z = sigmoid(z)
sigma_y = sigmoid(y)

plt.plot(y, sigma_y)
plt.axvline(0.0, color="k")  # pyright: ignore[reportArgumentType]
plt.ylim(-0.1, 1.1)
plt.xlabel("z")
plt.ylabel(r"$\sigma (z)$")
# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
plt.tight_layout()
plt.show()
