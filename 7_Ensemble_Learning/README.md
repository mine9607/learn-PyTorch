# Combining Different Models for Ensemble Learning

## Learning with ensembles

The goal of building ensemble models is to construct a set of classifiers that have a better predictive performance than any of the individual models

- Make predictions based on `majority voting`
- Use `bagging` to reduce overfitting (drawing random combinations of the training dataset with repetition)
- Apply `boosting` to build powerful models from `weak learners` that learn from their mistakes

`majority voting` - we select the class label that has been predicted by the majority of classifiers (> 50% of votes)

> NOTE: majority voting (in strict terms) refers to `binary classification` but can be extended to multi-class classification sometimes known as `plurality voting`

![Different voting concepts](./majority_vote.png)

> NOTE: Random Forest Classification is an example of building ensemble models

Depending on the technique, the ensemble can be built from:

1. Different classification algorithms (decision trees, SVMs, Logistic Regression, etc.) + Same Training Data
2. Different subsets of training data (nested k-fold CV) + Same Classifier

![General ensemble approach](./ensemble_approach.png)

We combine the predicted class labels of each individual classifier and select the class label that received the most votes (argmax)

## Combining classifiers via majority vote

Example of predicting majority vote on un-weighted and weighted classifiers

```python
# unweighted majority vote
print("Majority Vote:", np.argmax(np.bincount([0, 0, 1])))
# weighted majorty vote
print(
    "C3 Weighted Majority Vote:",
    np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])),
)
```

Remember using the `predict_proba_` method of some classifiers returns the class label predicted probabilities

Ex: 3 classfier ensemble model for binary classification (label 0 or 1)
Assumed classifier weights: wc1 = 0.2, wc2 = 0.2, wc3 = 0.6

The probabilities returned by the classifiers may be:
c1_0 = 0.9, c1_1 = 0.1, c2_0=0.8, c2_1 = 0.2, c3_0 = 0.4, c3_1=0.6

Therefore the weighted ensemble proabilities for class 0 and class 1 would be:

Class 0: $(0.2 * 0.9) + (0.2 * 0.8) + (0.6 * 0.4) = 0.58$
Class 1: $(0.2 * 0.1) + (0.2 * 0.2) + (0.6 * 0.6) = 0.42$

Argmax would then choose Class 0 (0.58)

### Implementing a simple majority vote classifier

See code

### Using the majority voting principle to make predictions

See code

### Evaluating and tuning the ensemble classifier

see code

## Bagging--building an ensemble of classifiers from bootstrap samples

`Bagging` (aka boostrap aggregating) - an ensemble learning technique closely related to `majority vote`

Instead of using the same training dataset to fit the individual classifiers in the ensemble, we draw `bootstrap samples` (random samples with replacement) - from the initial training dataset

### Bagging in a nutshell

1. Each classifier in the ensemble receives a random subset of the initial training dataset.
2. Fit each classifier to its sampled training data
3. Combine predictions from individual classifiers using majority voting

> NOTE: Random forests are a special case of bagging where we **ALSO** use random feature subsets when fitting the individual decision trees

### Applying bagging to classify examples in the Wine dataset

See code

## Leveraging weak learners via adaptive boosting

### How adaptive boosting works

### Applying AdaBoost using scikit-learn

## Gradient boosting--training an ensemble based on loss gradients

### Comparing AdaBoost with gradient boosting

### Outlining the general gradient boosting algorithm

### Explaining the gradient boosting algorithm for classification

### Illustrating gradient boosting for classification

### Using XGBoost

## Additional Resources:

- XGBoost: "https://xgboost.readthedocs.io/en/stable/"
- LightGBM: "https://lightgbm.readthedocs.io/en/lates/"
- CatBoost: "https://catboost.ai"
- HistGradientBoostingClassifier: "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html"
