# First Machine Learning Classifier

My first complete supervised machine learning project, built with **scikit-learn**. It covers both core problem types in one pipeline script:

1. **Linear Regression** - predicting California house prices
2. **Decision Tree Classifier** - classifying breast tumor diagnosis

## What the project demonstrates

- Loading built-in scikit-learn datasets
- `train_test_split` (80/20, with stratification for classification)
- Training models (`LinearRegression`, `DecisionTreeClassifier`)
- Evaluating with proper metrics: **MSE / RMSE / R-squared** for regression, **Accuracy / Confusion Matrix / Precision / Recall / F1** for classification
- Visualizing results (predictions scatter plot, confusion matrix, tree diagram)
- Comparing tree depth to find good `max_depth` instead of overfitting

## Getting started

```bash
pip install -r requirements.txt
python main.py
```

Charts are saved to `outputs/`.

## Part 1 - Linear Regression (California Housing)

- 20,640 samples, 8 features (median income, house age, location, ...)
- Target: median house value in $100,000s
- Test-set results:

| Metric | Value |
|---|---|
| MSE | 0.5559 |
| RMSE | 0.7456 |
| R-squared | 0.5758 |

Strongest positive coefficient: `MedInc` (median income), as expected.

## Part 2 - Decision Tree Classifier (Breast Cancer)

- 569 samples, 30 features, classes: malignant / benign
- Test-set results (`max_depth=4` chosen after comparing depth 2 to unlimited):

| Metric | Value |
|---|---|
| Accuracy | 0.9386 |
| Recall (malignant) | 0.9286 |
| F1 (macro avg) | 0.9343 |

Confusion matrix on the 114-sample test set:

|              | Pred. malignant | Pred. benign |
|---|---|---|
| Actual malignant | 39 | 3 |
| Actual benign | 4 | 68 |

## Project structure

```
main.py                  full pipeline (both parts)
requirements.txt         dependencies
outputs/
  regression_predictions.png   predictions vs actual prices
  confusion_matrix.png         decision tree confusion matrix
  decision_tree.png            the learned tree diagram
```

## Pipeline summary

```
load data -> train/test split -> fit model -> predict -> evaluate metrics
```
