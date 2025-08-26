import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from matplotlib.cbook import is_scalar_or_string
from pandas.io.stata import max_len_string_array
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

image_path = "./"
transform = transforms.Compose([transforms.ToTensor()])
"""
NOTE: ToTensor() converts the pixel features into a floating type tensort and ALSO normalizes the pixels from [0,255] to the range [0,1].  The labels are already integers from 0 to 9 so we don't need to do any scaling or conversion.
"""

# Original download
# mnist_train_dataset = torchvision.datasets.MNIST(
#     root=image_path, train=True, transform=transform, download=True
# )
# mnist_test_dataset = torchvision.datasets.MNIST(
#     root=image_path, train=False, transform=transform, download=True
# )

mnist_train_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True, transform=transform, download=False
)
mnist_test_dataset = torchvision.datasets.MNIST(
    root=image_path, train=False, transform=transform, download=False
)

batch_size = 64
torch.manual_seed(1)

"""
NOTE: a torch.utils.data.Dataset 

run: help(torchvision.datasets.MNIST)
"""
assert isinstance(mnist_train_dataset, torch.utils.data.Dataset)

# help(torchvision.datasets.MNIST)  # explore the helpful information about the dataset
# print(type(mnist_train_dataset))  # get dataset type
# print(dir(mnist_train_dataset))  # print list of methods and attributes

"""
NOTE... 

mnist_train_dataset.data && mnist_train_dataset.targets = full untouched dataset

mnist_train_dataset[0] = one processed sample (with transforms applied)
    - can only return one sample at a time at the specified index position (i.e. doesn't support slicing)  
    - returns (img, label) tuple

"""
# Look at raw tensors
print(mnist_train_dataset.data.shape)
print(mnist_train_dataset.targets.shape)

# Look at single example
img, label = mnist_train_dataset[0]
print(type(img), img.shape)
print(label)

train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)

## PREPROCESS THE IMAGES

hidden_units = [
    32,
    16,
]  # two hidden layers (one with 32 neurons and one with 16 neurons)
image_size = mnist_train_dataset[0][0].shape
print("Image Size: ", image_size)
# print(mnist_train_dataset[0])  # should return all image tensors
# print(mnist_train_dataset[0][0])  # should return the first image tensor
# print(mnist_train_dataset[1][0])  # should return the target of the first example

input_size = (
    image_size[0] * image_size[1] * image_size[2]
)  # (1, 28, 28) -- (channel, height, width)
all_layers = [
    nn.Flatten()
]  # list of layers - first layer flattens from (64, 1, 28, 28) -> (64, 784)
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit
all_layers.append(nn.Linear(hidden_units[-1], 10))
model = nn.Sequential(*all_layers)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
torch.manual_seed(1)
num_epochs = 20
for epoch in range(num_epochs):
    acc_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        acc_hist_train += is_correct.sum()
    acc_hist_train /= len(train_dl.dataset)
    print(f"Epoch {epoch} Accuracy {acc_hist_train:.4f}")

pred = model(mnist_test_dataset.data / 255.0)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f"Test accuracy: {is_correct.mean():.4f}")
