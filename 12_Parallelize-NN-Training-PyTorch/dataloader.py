import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

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
