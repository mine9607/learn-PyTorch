# Chapter 3

## A Tour of Machine Learning Classifiers Using Scikit-Learn

Choosing an appropriate classification algorithm for a particular problem task requires practice and experience.

**No Free Lunch Theorem**: no single classifier works best across all possible scenarios

> Note: It is always recommended that you compare the performance of at least a handful of different learning algorithms to select the best model for the particular problem

The five main steps involved in training a supervised machine learning algorithm can be summarized as:

1. Selecting features and collecting labeled training examples
2. Choosing a performance metric
3. Choosing a learning algorithm and training a model
4. Evaluating the performance of the model
5. Changing the settings of the algorithm and tuning the model

## Training a perceptron with Scikit-Learn

Using the Iris Dataset incorporated into the Scikit-Learn library, we see that the labels ("Iris-setosa, Iris-versicolor, Iris-virignica') are already encoded as integers (0,1,2)

This is a common best practice for class labels even if most scikit-learn functions and class methods work with strings--improves computational performance

### Data Splitting

We then split the dataset into training and test datasets.

> Note: The `train_test_split()` method shuffles the training dataset internally **BEFORE** splitting. If not the ordered data would have resulted in all examples from class 0 and class 1 in the training dataset and the test dataset would consist of 45 examples from class 2

> Note: `stratify=y` in the train_test_split method means that the train_test_split method returns training and test subsets that have the same proportions of class labels as the input dataset.

### Feature Scaling

It is important to scale features (0 to 1) for optimal performance and accuracy. We will use the `StandardScaler` class from scikit-learn

1. Import the StandardScaler class and instantiate a scaler
2. Use the `fit()` method to estimate the dataset distribution of the train data (mean and deviation)
3. Use th `transform()` method to normalize the train and test data using the estimated mean and standard deviation

### Train a Perceptron

1. Import and instantiate a `Perceptron` class instance
2. Instantiate the Perceptron instance with `eta0=0.01` and `random_state=1`
3. Fit the Perceptron to the training data using the `fit()` method: `ppn.fit(X_train_std, y_train)`
4. Predict the labels for the test inputs: `y_pred=ppn.predict(X_test)`
5. Calculate the misclassification percentage (error): `% (y_test != y_pred).sum())`

#### Classification Error vs Accuracy

> Instead of the misclassification error, many machine learning practitioners report the classification accuracy of a model, which is simply calculated as follows:
>
> 1-error

### Classification Metrics:

1.

### Logistic Regression

Logistic Regression can be used for `linear` and `binary` classification problems

> NOTE: Despite the name it is a classification model **NOT** a regression model

- performs well on linearly separable classes
- widely used for classification
- linear model for binary classification

> NOTE: logistic regression can be generalized to multiclass settings--`multinomial regression` aka `softmax regression`. Another way to use logistic regression in multiclass settings is via the OvR technique

The basis for the classification of an example via logistic regression relies on `odds` (i.e. $\frac{p}{(1-p)}$) where p is the probability that an example belongs to class 1 given its feature set.

The `logit function` is defined as the logarithm of the odds (log-odds):

$$
\text{logit}(p) = \log{\frac{p}{(1-p)}}
$$

The logit function taks values in the range 0 to 1 and transforms thme into values over the entire real number range

In logistic regression we **_ASSUME_** that there is a linear relationship between the `weighted inputs` ($w_ix_i$), aka net inputs and the `log-odds`

$$
\text{logit}(p) = w_1x_1 + ... + w_mx_m + b = \sum_{i=j}{w_jx_j} + b = w^Tx+b
$$

> If the logit function maps the probability to a real number range, the `inverse logit function` maps a real-number range back to a [0, 1] range for the probability (p)

The inverse logit function is called the `logistic sigmoid function` aka `sigmoid function` due to its characteristic S shape

$$
\sigma{(z)} = \frac{1}{1+e^{-z}}
$$

where z = net input = $w^Tx + b$

> NOTE: $\sigma{(z)} -> 1 $ as $z -> \inf$
