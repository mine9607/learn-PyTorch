# The Mechanics of PyTorch

## The Key features of PyTorch

1. PyTorch uses dynamic computational graphs, which are more flexible and debugger friendly than static competitors. With PyTorch you can execute the code line by line having full access to all variables

2. PyTorch has the ability to work with single or multiple GPUs--allowing efficient training on large datasets and large-scale systems

3. PyTorch supports mobile development, making it suitable for production.

## PyTorch's Computation Graphs

`Directed acyclic graph (DAG)` - used to derive relationships between tensors from the input all the way to the output.

Computation graphs assign nodes for each variable (tensor) and for each result of a mathematical operation.

![Simple Illustration of Computation Graph](./comp-graph.png)
