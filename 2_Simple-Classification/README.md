# Chapter 2

## Summary of Math of Perceptron

1. \*\*Net Input (z):\*
   The perceptron first computes the \*\*weighted sum\*\* of inputs:
   $z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
   This is the `net input`, sometimes written as $\mathbf{w} \cdot \mathbf{x} +b$, which will be passed to the model's activation/decision function
2. Decision Function ($\sigma(z)$):
   The decision function defines **how** the model makes predictions based on the net input z.

   - In a binary perceptron, this is often a step function
   - In more advanced models, decision functions include **sigmoid**, **softmax**, or **argmax** depending on the task

   > 📝 Note: In a single-layer perceptron--the decision function and activation function are typically the same.

3. Activation Function:
   In multi-layer networks, an `activation function` is applied at each neuron to introduce non-linearity (ReLU, Sigmoid, Tanh, Softmax, etc.)

   > 📝 Note: This enables deep networks to learn complex patterns beyond linear relationships

4. Optimization via Gradient Descent
   We use **gradient descent** (or a varient SGD or Adam) to update the weights **w** and bias **_b_**, minimizing a `loss function` like SSE or MSE

   - This is done by computing the gradient of the loss with respect to each parameter and taking steps in the negative direction

5. Prediction
   Once trained, the; perceptron takes new input vectors, computes the net input z, and applies the learned decision function $\sigma(z)$ to make predictions

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
1 & if z \geq \theta \\
0 & \text{otherwise}
\end{cases}
$$

So it is either "ON" or "OFF" (binary classification)

To simplify code implementation of this we move $\theta$ to the left side and add a _bias_ unit $\mathbf{b=-\theta}$.

$$
z >= \theta
$$

$$
z - \theta >=0
$$

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
w := w + \Delta w
$$

$$
b := b + \Delta b
$$

The changes are defined as the negative gradient multiplied by the learning rate $eta$

$$
\Delta w = -\eta\nabla_wL(w,b)
$$

$$
\Delta b = -\eta\nabla_bL(w,b)
$$

To compute the gradient of the loss function we need to compute the partial derivative of the loss function wrt each weight, $w_j$

$$
\frac{\partial L}{\partial w_j} = -\frac{2}{n}\sum_i (y^{(i)}-\sigma(z^{(i)}))x_j^{(i)}
$$

$$
\frac{\partial L}{\partial b} = -\frac{2}{n}\sum_i (y^{(i)}-\sigma(z^{(i)}))
$$

> NOTE: The 2 is just a scaling factor and can be omitted--doing so has the effect of changing the scaling rate by a factor of 2

We can then write the weight updates as:

$$
\Delta w_j = -\eta\frac{\partial L}{\partial w_j}
$$

$$
\Delta b = -\eta\frac{\partial L}{\partial b}
$$

The adaline learning rule becomes:

$$
w := w+ \Delta w
$$

$$
b:=b+\Delta b
$$

### Resulting Partial Derivatives

The result of the partial derivatives get evaluated to the following:

$$
\frac{\partial L}{\partial w_j} = -\frac{2}{n}\sum_i(y^{(i)}-\sigma(z^{(i)}))x_j^{(i)}
$$

$$
\frac{\partial L}{\partial b} = -\frac{2}{n}\sum_i(y^{(i)}-\sigma(z^{(i)}))
$$

> Although the adaline learning rule looks identical to the perceptron rule, $\sigma(z^{(i)})$ is a real number and not an integer class label.
>
> The weight update is calculated based on **ALL** examples in the training dataset (instead of updating incrementally after each training example)
>
> This approach is also referred to as **batch gradient descent** or **full batch gradient descent**

## Improving Gradient Descent: Feature Scaling

Gradient descent is one of the algorithms that benefits from feature scaling. We can use a normalization procedure called `standardization` to help GD converge more quickly.

> NOTE: Standardization does not make the dataset normally distributed. It simply shifts the mean of each feature so that it is centered at zero with standard deviation of 1.

### Standardization

Mathematically we standardize a feature by subtracting the sample mean and dividing by the standard deviation:

$$
x_j'=\frac{x_j - \mu_j}{\sigma_j}
$$

## Large-scale ML and Stochastic Gradient Descent

With full batch gradient descent we must evaluate the entire dataset each time we want to make a step towards the global minimum which is computationally expensive on very large datasets (millions of data points). To improve on this it is common to implement `Stochastic Gradient Descent (SGD)`. Instead of updating the weights based on the sum of the accumulated errors over ALL training examples, we update the parameters incrementally for each training example:

Full Scale GD:

$$
\Delta w_j = \frac{2\eta}{n} \sum_i (y^{(i)} - \sigma (z^{(i)}))x_j^{(i)}
$$

Stochastic GD:

$$
\Delta w_j = \eta(y^{(i)} - \sigma (z^{(i)}))x_j^{(i)}
$$

$$
\Delta b = \eta(y^{(i)}-\sigma(z^{(i)}))
$$

It is considered an approximation to Full Scale GD but reaches convergence much faster since weights are updated more frequently.

NOTE: Since each gradient is calculated based on a single training example, the error surface is noisier than in GD. This can have the advantage that SGD can escape shallow local minima more readily if working with nonlinear loss functions

To obtain good results with SGD it is important to present training data in a `random order`--we want to shuffle the training dataset for every epoch to prevent cycles.

> **Adjusting Learning Rate During Training:** In SGD, the fixed LR $\eta$ is often replaced by an adaptive learning rate that decreases over time
>
> $$
> \frac{c_1}{[number of iterations]+c_2}
> $$
>
> where $c_1$ and $c_2$ are constants

SGD also allows for **online learning**, where our model is trained on the fly as new data arrives. Using online learning the system can immediately adapt to changes, and the training data can be discarded after updating the model if storage space is an issue.

> ** Mini-batch Gradient Descent**
> Compromise between full-batch GD and SGD
>
> Applies full-batch GD to a smaller subset of the training data (e.g. 32 training examples at a time)
>
> Converges faster than full-batch GD and allows us to replace the for loop over the training datat in SGD with vectorized operations improving computational efficiency.

# SUMMARY OF STEPS

## Full-Batch Gradient Descent (with Sigmoid + MSE Loss)

### 1. Initialize Parameters

- Randomly initialize weights `w_1, w_2, ..., w_d`
- Initialize bias `b`

---

### 2. For Each Epoch

#### a. Forward Pass

- For each training example `x^i`, compute:

$$
z^{(i)} = w^T * x^{(i)} + b
$$

$$
\hat{y}^{(i)} = \sigma(z^{(i)})
$$

#### b. Compute Loss

- Use the Mean Squared Error (MSE) loss:

$$
L = \frac{1}{n} * \sum_{i=1}^n(y^{(i)} - \hat{y}^{(i)})^2
$$

#### c. Backward Pass (Compute Gradients)

- For each weight `w_j`, compute the average gradient over all examples:

$$
\frac{\partial{L}}{\partial{w_j}} = \frac{1}{n} \sum_i \frac{\partial{L^{(i)}}}{\partial{w_j}} = -\frac{2}{n}\sum_i(y^{(i)}-\hat{y}^{(i)})x_j^{(i)}
$$

$$
\frac{\partial{L}}{\partial{b}} = \frac{1}{n} \sum_i\frac{\partial{L}^{(i)}}{\partial{b}} = -\frac{2}{n}\sum_i(y^{(i)}-\hat{y}^{(i)})
$$

#### d. Gradient Descent Update

- Update weights and bias:

$$
w_j = w_j - \eta \frac{\partial{L}}{\partial{w_j}}
$$

$$
b = b - \eta \frac{\partial{L}}{\partial{b}}
$$

---

### 3. Repeat

- Continue until convergence or maximum number of epochs is reached.

---

## Stochastic Gradient Descent (SGD) — Sigmoid + MSE Loss

### 1. Initialize Parameters

- Randomly initialize weights `w_1, w_2, ..., w_d`
- Initialize bias `b`

---

### 2. For Each Epoch

- Shuffle the training data

- For each training example `(x^(i), y^(i))`:

  #### a. Forward Pass

$$
z^{(i)} = w^T * x^{(i)} + b
$$

$$
\hat{y}^{(i)} = \sigma(z^{(i)})
$$

#### b. Compute Instantaneous Squared Error

- Compute the **squared error** for this example:

$$
L^{(i)} = (y^{(i)}-\hat{y}^{(i)})^2
$$

#### c. Compute Gradients

- For each weight `w_j` and the bias `b`, compute the gradients using only this one data point:

$$
\frac{\partial{L^{(i)}}}{\partial{w_j}}
$$

$$
\frac{\partial{L^{(i)}}}{\partial{b}}
$$

#### d. Update Parameters

$$
w_j = w_j - \eta \frac{\partial{L^{(i)}}}{\partial{w_j}}
$$

$$
b = b - \eta \frac{\partial{L^{(i)}}}{\partial{b}}
$$

---

### 3. Repeat

- Continue looping through examples across epochs until convergence or a max number of epochs is reached.
