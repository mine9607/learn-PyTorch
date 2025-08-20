# Parallelizing Neural Network Training with PyTorch

## PyTorch and training performance

### Performance challenges

### What is PyTorch

PyTorch is a scalable and multi-platform programming interface for implementing and running machine learning algorithms.

PyTorch allows execution on CPUs, GPUs and XLA devices (TPUs).

`Tensors` can be understood as a generalization of `scalars`, `vectors`, `matrices`, etc.

- A `scalar` is a rank-0 tensor

- A `vector` is a rank-1 tensor

- A `matrix` is a rank-2 tensor

- `Matrices` can be thought of as "stacked matrixes" (in 3D) this is a rank-3 tensor

![Different types of tensor in PyTorch](./tensors.png)

## First Steps with PyTorch:

### Installing PyTorch

### Creating `tensors` in PyTorch

### Manipulating data type and shape of a tensor

### Applying mathematical operations to tensors

### Split, stack, and concatenate tensors

## Building Input Pipelines in PyTorch:

### Creating a PyTorch DataLoader from existing tensors

### Combining two tensors into a joint dataset

### `Shuffle`, `batch`, and `repeat`

### Creating a dataset from files on local storage disk

### Fetching available datasets from the `torchvision.datasets` library

## Building a NN model in PyTorch

### The PyTorch NN module (`torch.nn`)

### Building a linear regression model

### Model training via the torch.nn and torch.optim modules

### Building a multilayer perceptron for classifying flowers in the Iris dataset

### Evaluating the trained model on the test dataset

### Saving and reloading the trained Model

## Choosing activation functions for multi-layer neural networks (MLNNs)

### Logistic function recap

### Estimating class probabilities in multiclass classification via the `softmax` function

### Broadening the output spectrum using a hyperbolic tangent

### Rectified Linear Unit (ReLU) activation
