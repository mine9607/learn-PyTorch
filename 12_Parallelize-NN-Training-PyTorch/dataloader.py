import os
import pathlib
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import CelebA

t = torch.arange(6, dtype=torch.float32)
data_loader = DataLoader(t)

# Iterate through dataset
for item in data_loader:
    print(item)

# Create batches of size 3
data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    print(f"batch {i}:", batch)

# Combining two tensors into a dataset

torch.manual_seed(1)
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)


class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


"""
NOTE: a custom Dataset must implement __init__() and __getitem__() methods
"""

# joint_dataset = JointDataset(t_x, t_y)  # using our class implementation above
joint_dataset = TensorDataset(t_x, t_y)  # using torch.utils.data.TensorDataset

for example in joint_dataset:
    print(" x: ", example[0], " y: ", example[1])

#### SHUFFLE DATASET ####
# Create a shuffled dataset from the DataLoader
torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)
for i, batch in enumerate(data_loader, 1):
    print(f"batch {i}:", "\nx:", batch[0], "\ny:", batch[1])

"""
NOTE: when training a model for multiple epochs we need to shuffle and iterate over the dataset by the desired number of epochs
"""

for epoch in range(2):
    print(f"epoch {epoch+1}")
    for i, batch in enumerate(data_loader, 1):
        print(f"batch {i}:", "\nx:", batch[0], "\ny:", batch[1])


#### CREATE DATASET FROM LOCAL FILES ####

# generate a list of files
imgdir_path = pathlib.Path("cat_dog_images")
file_list = sorted([str(path) for path in imgdir_path.glob("*.jpg")])
print(file_list)

# Visualize image examples
fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img = Image.open(file)
    print("Image shape:", np.array(img).shape)
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)

plt.tight_layout()
plt.show()

# extract class labels from list of filenames
labels = [1 if "dog" in os.path.basename(file) else 0 for file in file_list]
print(labels)


# Create a joint dataset of data and labels
class ImageDataset1(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels

    def __getitem__(self, idx):
        file = self.file_list[idx]
        label = self.labels[idx]
        return file, label

    def __len__(self):
        return len(self.labels)


image_dataset = ImageDataset1(file_list, labels)

for file, label in image_dataset:
    print(file, label)

# Apply transformations to the dataset
"""
1. Load the image content from its file path
2. Decode the raw content
3. Resize the image to a desired (standard) size - (80x120)
"""


class ImageDatasetTransform(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.labels)


img_height, img_width = 80, 120
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (img_height, img_width),
        ),
    ]
)

image_dataset = ImageDatasetTransform(file_list, labels, transform)

fig = plt.figure(figsize=(10, 6))
for i, example in enumerate(image_dataset):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0].numpy().transpose((1, 2, 0)))
    ax.set_title(f"{example[1]}", size=15)

plt.tight_layout()
plt.show()

#### Get CelebA DAtaset ####

image_path = "./"
celeba_dataset = CelebA(image_path, split="train", target_type="attr", download=True)

assert isinstance(celeba_dataset, torch.utils.data.Dataset)

example = next(iter(celeba_dataset))
print(example)

fig = plt.figure(figsize=(12, 8))
for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
    ax = fig.add_subplot(3, 6, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f"{attributes[31]}", size=15)

plt.show()

# Download the MNIST dataset
# mnist_dataset = torchvision.datasets.MNIST(image_path, "train", download=True)
mnist_dataset = torchvision.datasets.MNIST(image_path, "train", download=False)

assert isinstance(mnist_dataset, torch.utils.data.Dataset)
example = next(iter(mnist_dataset))

print(example)

fig = plt.figure(figsize=(15, 6))
for i, (image, label) in islice(enumerate(mnist_dataset), 10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image, cmap="gray_r")
    ax.set_title(f"{label}", size=15)

plt.show()


#### BUILDING A NN MODEL IN PYTORCH ####

# Step 1 - Create a dataset
X_train = np.arange(10, dtype="float32").reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype="float32")
plt.plot(X_train, y_train, "o", markersize=10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Step 2 - Standardize the Data and create a TensorDataset
sc = StandardScaler()
X_train_norm = sc.fit_transform(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train).float()

train_ds = TensorDataset(X_train_norm, y_train)

batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Step 3 - Define model parameters

torch.manual_seed(1)

weight = torch.randn(1)
weight.requires_grad_()
bias = torch.zeros(1, requires_grad=True)


def model(xb):
    return xb @ weight + bias  # @ is matrix multiplication in python


# Step 4 - Define the Loss function - MSE
def loss_fn(input, target):

    return (input - target).pow(2).mean()


learning_rate = 0.001
num_epochs = 200
log_epochs = 10
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch.long())
        loss.backward()
    with torch.no_grad():
        weight -= weight.grad * learning_rate
        bias -= bias.grad * learning_rate
        weight.grad.zero_()
        bias.grad.zero_()
    if epoch % log_epochs:
        print(f"Epoch: {epoch} Loss {loss.item():.4f}")


print("Final Parameters:", weight.item(), bias.item())

X_test = np.linspace(0, 9, num=100, dtype="float32").reshape(-1, 1)
X_test_norm = sc.transform(X_test)
X_test_norm = torch.from_numpy(X_test_norm)

y_pred = model(X_test_norm).detach().numpy()

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, "o", markersize=10)
plt.plot(X_test_norm, y_pred, "--", lw=3)
plt.legend(["Training Examples", "Linear reg."], fontsize=15)
ax.set_xlabel("x", size=15)
ax.set_ylabel("y", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)
plt.show()

# Model training via torch.nn and toch.optim modules

loss_fn = nn.MSELoss(reduction="mean")
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # 1. Generate predictions
        pred = model(x_batch)[:, 0]
        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)
        # 3. Compute gradients
        loss.backward()
        # 4. Update parameters from gradients
        optimizer.step()
        # 5. Reset gradients to zero for next iteration
        optimizer.zero_grad()
    if epoch % log_epochs == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")

print("Final Parameters:", model.weight.item(), model.bias.item())

# Build a multilayer perceptron for classifying flowers in Iris Dataset

iris = load_iris()
X = iris["data"]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1.0 / 3, random_state=1
)

X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)

X_train_norm = torch.from_numpy(X_train_norm).float()
X_test_norm = torch.from_numpy(X_test_norm).float()

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

train_ds = TensorDataset(X_train_norm, y_train)
torch.manual_seed(1)

batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)

        return x


input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3

model = Model(input_size, hidden_size, output_size)

learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100
loss_hist = [0] * num_epochs
acc_hist = [0] * num_epochs

for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist[epoch] += loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        acc_hist[epoch] += is_correct.sum()
    loss_hist[epoch] /= len(train_dl.dataset)
    acc_hist[epoch] /= len(train_dl.dataset)


fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)
ax.set_title("Training Loss", size=15)
ax.set_xlabel("Epoch", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(acc_hist, lw=3)
ax.set_title("Training Accuracy", size=15)
ax.set_xlabel("Epoch", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)

plt.show()

# Evaluating the trained model on the test datset

pred_test = model(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()

accuracy = correct.mean()
print(f"Test Accuracy: {accuracy:.4f}")

# Saving and reloading the trained model
path = "iris_classifier.pt"
torch.save(model, path)

model_new = torch.load(path)
model_new.eval()

pred_test = model_new(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
accuracy = correct.mean()
print(f"Test Acc: {accuracy:.4f}")
