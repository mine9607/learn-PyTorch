import torch
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
#### BATCH DATASET ####


#### REPEAT DATASET ####
