# Learning Best Practices for Model Evaluation and Hyperparameter Tuning

In this chapter we learn how to:

1. Assess the performance of machine learning models
2. Diagnose the common problems of machine learning algorithms
3. Fine-tune machine learning models
4. Evaluate predictive models using different performance metrics

## Streamline workflows with pipelines

We can use the `Pipeline Class` from scikit-learn which allows us to fit a model including any arbitrary number of transformation steps and apply it to make predictions about new data

## Using k-fold cross-validation to assess model performance

Cross-validation is used to help obtain reliable estimates of the model's generalization performance

### 1. Holdout Cross-Validation

In the `holdout method` we split the initial dataset into separate training and test datasets--however, we are also interested in tuning and comparing differnt parameter settings (`hyperparameters`). This process is called `model selection`.

If we reuse the same test dataset over and over again during model selection, it will become part of our training data and thus the model will be more likely to overfit. Many people still use the test dataset for `model selection` which is not a good machine learning practice.

> NOTE: This is very important to understand. Any time we are performing `model selection` by fine-tuning hyperparameters--if we use the `test dataset` to check the current model's accuracy and then update hyperparameters based on the results on the test data--we are effectively `leaking` the test data into the training process. We should therefore ssplit the data into a third portion (training/validation/test) so that the test dataset is truly unseen and only used 1 time when the hyperparameters have been locked (optimized based on the validation set). If we go back and tune the model after seeing poor results on the test dataset, we have corrupted the model again.

- `training data` - used to fit the different models

- `validation data` - used to evaluate model performance and tune hyperparameters (`model selection`)

- `test data` - used to estimate the model's ability to generalize to unseen data

![Holdout Cross-Validation vs Nested-Cross Validation](./output.png)

A **DISADVANTAGE** of the holdout method is that the performance estimate may be very sensitive to **_HOW_** we partition the training dataset into the `training` and `validation` subsets.

### 2. K-fold Cross-Validation

In `k-fold cross-validation` we **RANDOMLY** split the `training dataset` into k folds without replacement. Here, k-1 folds (`training folds`), are used for the model training and 1 fold (`test fold`), is used for performance evaluation. This is repeated k times so that we obtain k models and performance estimates.

> NOTE: Reminder--`random selection without replacement`--you **NEVER** pick the same example twice in the same draw, once an item is chosen, it is removed from the "pool". E.g. drawing 5 cards from a deck of 52

> NOTE: Reminder--`random selection with replacement`--after you pick an example, you put it `back into` the pool before the next draw. The same example **CAN BE** picked multiple times. E.g. rolling a die 10 times

We calculate the `average performance` of the models based on the different, independent test folds to obtain a performance estimate that is `less sensitive` to the sub-partitioning of the training data (compared to `holdout method`).

#### Steps in k-fold cross-validation

Train/holdout split -> CV on training set

1. Split data into a large training set and a final untouched holdout set (test)
2. Use k-fold CV **only on the training set** to pick the best hyperparameters
3. Retrain with the best hyperparameters on **ALL** training data
4. Evaluate model **ONCE** on the untouched holdout data (test)

The advantage is that in each iteration, each example will be used exactly once and the training and test folds are dijoint (no overlap)

![How k-fold cross-validation works](./k-folds.png)
