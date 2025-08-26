import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

fig = plt.figure(figsize=(6, 6))
plt.plot(x[y == 0, 0], x[y == 0, 1], "o", alpha=0.75, markersize=10)
plt.plot(x[y == 1, 0], x[y == 1, 1], "<", alpha=0.75, markersize=10)
plt.xlabel(r"$x_1$", size=15)
plt.ylabel(r"$x_2$", size=15)
plt.show()

#### CReate a baseline simple model
base_model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid(),
)

print(base_model)

#### Initialize optimizer and loss function

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(base_model.parameters(), lr=0.001)

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
            acc_hist_train[epoch] += is_correct.mean()
        loss_hist_train[epoch] /= n_train / batch_size
        acc_hist_train[epoch] /= n_train / batch_size
        pred = model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred >= 0.5).float() == y_valid).float()
        acc_hist_valid[epoch] += is_correct.mean()
    return loss_hist_train, loss_hist_valid, acc_hist_train, acc_hist_valid


history = train(base_model, num_epochs, train_dl, x_valid, y_valid)

#### Plot the results

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history[0], lw=4)
plt.plot(history[1], lw=4)
plt.legend(["Train Loss", "Validation Loss"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(history[2], lw=4)
plt.plot(history[3], lw=4)
plt.legend(["Train Acc:", "Validation Acc:"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
plt.show()

#### Model 2: 1 hidden layer - 4 neurons
model2 = nn.Sequential(
    nn.Linear(2, 4),  # input layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 1st hidden layer
    nn.ReLU(),
    nn.Linear(4, 1),  # output layer
    nn.Sigmoid(),
)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)
history2 = train(model2, num_epochs, train_dl, x_valid, y_valid)

#### Model 3: 2 hidden layer - 4 neurons
model3 = nn.Sequential(
    nn.Linear(2, 4),  # input layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 1st hidden layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 2nd hidden layer
    nn.ReLU(),
    nn.Linear(4, 1),  # output layer
    nn.Sigmoid(),
)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model3.parameters(), lr=0.01)
history3 = train(model3, num_epochs, train_dl, x_valid, y_valid)

#### Model 4: 3 hidden layers - 4 neurons
model4 = nn.Sequential(
    nn.Linear(2, 4),  # input layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 1st hidden layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 2nd hidden layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 3rd hidden layer
    nn.ReLU(),
    nn.Linear(4, 1),  # output layer
    nn.Sigmoid(),
)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model4.parameters(), lr=0.01)
history4 = train(model4, num_epochs, train_dl, x_valid, y_valid)

#### Model 5: 4 hidden layers - 4 neurons
model5 = nn.Sequential(
    nn.Linear(2, 4),  # input layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 1st hidden layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 2nd hidden layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 3rd hidden layer
    nn.ReLU(),
    nn.Linear(4, 4),  # 4th hidden layer
    nn.ReLU(),
    nn.Linear(4, 1),  # output layer
    nn.Sigmoid(),
)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model5.parameters(), lr=0.01)
history5 = train(model5, num_epochs, train_dl, x_valid, y_valid)

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.title("1 Hidden Layer Model")
plt.plot(history2[0], lw=4)
plt.plot(history2[1], lw=4)
plt.legend(["Train Loss", "Validation Loss"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(history2[2], lw=4)
plt.plot(history2[3], lw=4)
plt.legend(["Train Acc:", "Validation Acc:"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
plt.show()

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.title("2 Hidden Layer Model")
plt.plot(history3[0], lw=4)
plt.plot(history3[1], lw=4)
plt.legend(["Train Loss", "Validation Loss"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(history3[2], lw=4)
plt.plot(history3[3], lw=4)
plt.legend(["Train Acc:", "Validation Acc:"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
plt.show()


fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.title("3 Hidden Layer Model")
plt.plot(history4[0], lw=4)
plt.plot(history4[1], lw=4)
plt.legend(["Train Loss", "Validation Loss"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(history4[2], lw=4)
plt.plot(history4[3], lw=4)
plt.legend(["Train Acc:", "Validation Acc:"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
plt.show()


fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.title("4 Hidden Layer Model")
plt.plot(history5[0], lw=4)
plt.plot(history5[1], lw=4)
plt.legend(["Train Loss", "Validation Loss"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(history5[2], lw=4)
plt.plot(history5[3], lw=4)
plt.legend(["Train Acc:", "Validation Acc:"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
plt.show()
