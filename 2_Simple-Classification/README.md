# Chapter 2

## Training Simple Machine Learning Algorithms for Classification

- Build and understand machine learning algorithms
- Use pandas, numpy, and matplotlib to read in, process and visualize data
- Implementing linear classifiers for 2-class problems in Python

## Formal Definition: Artificial Neuron

$$
z = w_1x_1 + w_2x_2 + ...+ w_mx_m
$$

In the perceptron algorithm, the decision function $\sigma$ is a variant of a unit step function:

$$
\sigma(z) =
\begin{cases}
1 & if z \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

So it is either "ON" or "OFF" (binary classification)

To simplify code implementation of this we move $\theta$ to the left side and add a _bias_ unit $\mathbf{b=-\theta}$.

$$
z = w_1x_1 + ... + w_mx_m + b = w^Tx + b
$$

### Linear Algebra: Dot Product and Matrix Transpose

We use the vector dot product to abbreviate the sum of the products of the values in x and w:

$$
a =
\begin{bmatrix}
a_1 \\
a_2 \\
a_3
\end{bmatrix},\quad
b =
\begin{bmatrix}
b_1 \\
b_2 \\
b_3
\end{bmatrix}
$$

The Transpose operation transforms a column vector into a row vector and vice versa

$$
a^T b = \sum_i a_i b_i = a_1 \cdot b_1 + a_2 \cdot b_2 + a_3 \cdot b_3
$$

## Perceptron Learning Rule

1. Initialize the weights and bias unit to 0 or small random numbers
2. For each training example $x^{(i)}$:

   a - Compute the output value, $\hat{y}^{(i)}$

   b - Update the weights and bias unit

Here the output value is the class label predicted by the unit step function defined above and the simultaneous update
of the bias unit and each weight $w_j$, in the weight vector **w**

$$
w_j := w_j + \Delta w_j
$$

$$
b := b + \Delta b
$$

The update values (deltas) are computed as:

$$
\Delta w_j = \eta \left(y^{(i)} - \hat{y}^{(i)}\right) x_j^{(i)}
$$

$$
\Delta b = \eta \left(y^{(i)} - \hat{y}^{(i)}\right)
$$

Each weight $w_j$ corresponds to a feature $x_j$ in the dataset

$\eta$ is the `learning_rate` (typically between 0 and 1)

$y^{(i)}$ is the `true class label`

$\hat{y}^{(i)}$ is the `predicted class label`

> Note: the convergence of the perceptron is ONLY guaranteed if the two classes are linearly separable (due to the cost
> function `decision` function being defined as unit step function (0 or 1) in this case

## Adaptive Linear Neurons and the Convergence of Learning

`ADAptive LInear NEuron` (Adaline) - improves upon the simple perceptron

In the adaline rule the weights are updated based on a linear activation function rather than a unit step function.

In Adaline, the linear activation function $\sigma(z) = z$

In this type of neuron the linear activation function is used to learn the weights but a threshold function (such as unit step) is still used to make the final prediction

## Minimizing Loss Functions with Gradient Descent

One key of supervised learning is a well-defined **objective function** or loss function to be optimized during the learning process.

> NOTE: Think about minimizing the SSE of a prediction

In the case of Adaline, we define the loss function L, to learn the model parameters as the **mean squared error (MSE)** between the calculated outcome and the true class label

$$
L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \sigma(z^{(i)}))^2
$$

There are important features of this loss function L:

1. It is continuous and therefore differentiable.
2. It is convex--allows us to use a powerful optimization algorithm called **gradient descent** to find the weights (minimize the loss function)

### Gradient Descent:

In each step or iteration we take a step in the direction **OPPOSITE** to the gradient where the step size is determined by the value of the `learning rate` as well as the `slope of the gradient`

This can be written as:

$$
w := w + \delta{w}

b := b + \delta{b}
$$

