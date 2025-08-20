# Implementing a Multi-layer Artificial Neural Network (ANN) from Scratch

Deep learning can be thought of as a sub-field of Machine Learning (ML) that is concerned with training `artificial neural networks` (ANNs).

## Modeling complex functions with ANNs

Artificial neurons represent the building blocks of multi-layer ANNs.

## Single-layer Neural Network Recap

`Adaptive Linear Neuron (Adaline)`: we have seen how this can be used for binary classification and used the gradient descent optimization algorithm to learn the weights of the model. In each epoch we updated the weight vector (w) and bias unit (b).

Recall that we computed the gradient based on the whole training dataset and updated the weights of the model by taking a step opposite of the loss gradient. To find the optimal weights we optimized an objective function (MSE) and we multiplied the gradient by a factor--the learning rate ($\eta$)

In `gradient descent` we updated **ALL** the weights simultaneously after each `epoch`. We derived the partial derivative of the weights and bias w.r.t. the Loss function and applied an `activation function` (linear for adaline). Finally, we implemented a `threshold function` to squash the continuous-valued output into binary class labels (0 or 1).

We also covered `stochastic gradient descent` which approximates the loss from a single training sample (online learning) or a small subset of the training examples (`mini batch learning`).

> NOTE: the noise introduced by SGD helps multilayer NNs with nonlinear activation functions which do not have a convex loss function by helping to escape local minima and converge to a more optimal solution.

### Introducing the Multi-layer NN Architecture

A `multi-layer neural network` (MLP) is a special type of `fully connected` network

> NOTE: we can think of the number of hidden layers in a NN as another hyperparameter that we want to optimize for by using `cross validation`

The loss gradients for updating the network's parameters--via backpropagation--will become increasingly small as more layers are added. This vanishing gradient (loss of information needed to optimize) problem makes model learning more challenging.

### Activating a NN via forward propagation

The process of `forward propagation` is used to calculate the output of an MLP model.

1. Starting at the input layer--forward propagate the patterns of the training data through the network to generate an output.

2. Based on the network's output, calculate the Loss that we want to minimize using a `loss function`

3. Backpropagate the Loss--find its derivative w.r.t each weigth and bias unit in the network and update the model

4. Repeat for multiple epochs

5. Finally, use forward propagate the learned weights and bias units to compute the network's output and apply a `threshold function` to obtain the predicted class labels in the one-hot representation.

Each unit in the hidden layer is connected to **ALL** units in the input layer we calculate the activation unit of the hidden layer

The activation function has to be differentiable to learn the weights that connect the neurons using gradient approach (SGD). To solve complex problems we need a nonlinear activation function such as the sigmoid (logistic) activation function.

> NOTE: MLP is an example of a `feed forward` ANN. `feedforward` refers to the fact that each layer serves as the input to the next layer (without loops)--in contrast to recurrent NN (RNNs)

## Classifying handwritten digits

### Obtaining and preparing the MNIST dataset

## Implementing a multi-layer perceptron

### Coding the NN training loop

## Evaluating the NN performance

## Training an ANN

### Computing the Loss Function

### Developing an understanding of backpropagation

### Training NN via backpropagation

## About the Convergence in NN
