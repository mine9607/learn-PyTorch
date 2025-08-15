# Predicting Continous Target Variables wih Regression Analysis

Regression models are a form of supervised learning which are used to predict target variables on a continuous scale.

They are used for:

- understanding relationships between variables
- evaluating trends
- making forecasts

In this chapter we discuss the following topics:

- Exploring and visualizing datasets
- Looking at different approaches to implementing linear regression models
- Training regression models that are robust to outliers
- Evaluating regression models and diagnosing common problems

## Introducing Linear Regression

### Simple Linear Regression

In simple linear regression we try to find the best straight line fit to the dataset given a single feature (y=mx+b)

$$
y = w_1x+ b
$$

The best fitting line is called the **regression line** and the distance from the line to the training examples are the **residuals**--the error of our prediction

### Multiple Linear Regression

Same as with simple linear regression but here we have more than one feature variable and hence more than one weight

$$
y = w_1x_1 + ...+w_mx_m + b = \sum_{i=1}^m w_ix_i +b = w^Tx + b
$$

## Exploring the Ames Housing Dataset

### Loading the Data into a DataFrame

### Visualizing the Data

### Correlation Matrix

## Implementing an ordinary least squares linear regression model

### Solving regression for regression parameters with gradient descent

### Estimating the coefficient of a regression model

## Fitting a robust regression model using RANSAC

## Evaluating the performance of Linear Regression models

### Residual Plots

### Mean Square Error

### Mean Average Error

### Coefficient of Determination

## Using regularized methods for regression

### Ridge Regression

### Least Absolute Shrinkage and Selection Optimizer (LASSO)

### Elastic Net

## Turning a Linear Regression model into a Curve: Polynomial Regression

### Modeling nonlinear relationships in the Ames Dataset

## Dealing with non-linear relationships using random forests

### Decision Tree Regression

### Random Forest Regression
