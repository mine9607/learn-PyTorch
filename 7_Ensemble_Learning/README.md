# Combining Different Models for Ensemble Learning

## Learning with ensembles

The goal of building ensemble models is to construct a set of classifiers that have a better predictive performance than any of the individual models

- Make predictions based on `majority voting`
- Use `bagging` to reduce overfitting (drawing random combinations of the training dataset with repetition)
- Apply `boosting` to build powerful models from `weak learners` that learn from their mistakes

`majority voting` - we select the class label that has been predicted by the majority of classifiers (> 50% of votes)

> NOTE: majority voting (in strict terms) refers to `binary classification` but can be extended to multi-class classification sometimes known as `plurality voting`

![Different voting concepts](./majority_vote.png)

## Combining classifiers via majority vote

### Implementing a simple majority vote classifier

### Using the majority voting principle to make predictions

### Evaluating and tuning the ensemble classifier

## Bagging--building an ensemble of classifiers from bootstrap samples

### Bagging in a nutshell

### Applying bagging to classify examples in the Wine dataset

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
