import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mlxtend.plotting import plot_decision_regions
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)


def compute_z(a, b, c):
    r1 = torch.sub(a, b)
    r2 = torch.mul(r1, 2)
    z = torch.add(r2, c)
    return z


print("Scalar Inputs:", compute_z(torch.tensor(1), torch.tensor(2), torch.tensor(3)))

print(
    "Rank 1 Inputs:", compute_z(torch.tensor([1]), torch.tensor([2]), torch.tensor([3]))
)

print(
    "Rank 2 Inputs:",
    compute_z(torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[3]])),
)

a = torch.tensor(3.14, requires_grad=True)
print(a)
b = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(b)

# note requires_grad = False by default and can more efficiently be set to true by the .requires_grad_() method

w = torch.tensor([1.0, 2.0, 3.0])
print(w.requires_grad)

w.requires_grad_()
print(w.requires_grad)

w = torch.empty([2, 3])
print(w)
nn.init.xavier_normal_(w)
print(w)

# Define two Tensor-objects inside the base nn.Module class


# class MyModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.w1 = torch.empty(2, 3, requires_grad=True)
#         nn.init.xavier_normal_(self.w1)
#         self.w2 = torch.empty(1, 2, requires_grad=True)
#         nn.init.xavier_normal_(self.w2)
#

# these two weights tensors are now initialized and enabled for gradient computation (backprop)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
x = torch.tensor([1.4])
y = torch.tensor([2.1])
z = torch.add(torch.mul(w, x), b)
loss = (
    (y - z).pow(2).sum()
)  # loss function (partial derivative of sigmoid activation function)
loss.backward()
print("dL/dw: ", w.grad)
print("dL/db: ", b.grad)

# verify the computed gradient
print(2 * x * ((w * x + b) - y))

# Build a 2-layer fully-connected model using nn.Sequential
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
)

print(model)

# configure the first layer by specifying the initial value distribution for the weight
nn.init.xavier_uniform_(model[0].weight)

# configure the second layer by computing the l1 penalty term for the weight matrix:
l1_weight = 0.01
l1_penalty = l1_weight * model[2].weight.abs().sum()

# Use SGD optimization with Binary cross-entropy loss function for binary classification
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Solving an XOR classification problem
torch.manual_seed(1)
np.random.seed(1)

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))

y[x[:, 0] * x[:, 1] < 0] = 0
n_train = 100
x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.float32)
x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
y_valid = torch.tensor(y[n_train:], dtype=torch.float32)

#### Create a data loader to batch the inputs
train_ds = TensorDataset(x_train, y_train)
batch_size = 2
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

#### Train the base_model
torch.manual_seed(1)
num_epochs = 200


def train(model, num_epochs, train_dl, x_valid, y_valid):
    loss_hist_train = [0] * num_epochs
    acc_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    acc_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            acc_hist_train[epoch] += is_correct.mean().item()
        loss_hist_train[epoch] /= n_train / batch_size
        acc_hist_train[epoch] /= n_train / batch_size
        pred = model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred >= 0.5).float() == y_valid).float()
        acc_hist_valid[epoch] = is_correct.mean().item()
    return loss_hist_train, loss_hist_valid, acc_hist_train, acc_hist_valid


# CREATE A CUSTOM LAYER TO ADD NOISE TO DATA
class NoisyLinear(nn.Module):
    def __init__(self, input_size, output_size, noise_stddev=0.1):
        super().__init__()
        w = torch.Tensor(input_size, output_size)
        self.w = nn.Parameter(w)  # nn.Parameter is a Tensor that's a module parameter
        nn.init.xavier_uniform_(self.w)
        b = torch.Tensor(output_size).fill_(0)
        self.b = nn.Parameter(b)
        self.noise_stddev = noise_stddev

    def forward(self, x, training=False):
        if training:
            noise = torch.normal(0.0, self.noise_stddev, x.shape)
            x_new = torch.add(x, noise)
        else:
            x_new = x
        return torch.add(torch.mm(x_new, self.w), self.b)


# Test the layer

torch.manual_seed(1)
noisy_layer = NoisyLinear(4, 2)
x = torch.zeros((1, 4))
print(noisy_layer(x, training=True))

print(noisy_layer(x, training=True))

print(noisy_layer(x, training=False))
