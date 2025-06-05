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

To simplify code implementation of this we move $\theta$ to the left side and add a _bias_ unit \*\*$b=-\theta$\*\*
