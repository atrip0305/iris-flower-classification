# iris-flower-classification
## Project description
---

# Iris Dataset Flower Classification

This project demonstrates a machine learning workflow for classifying iris flowers into three species: **Setosa**, **Versicolor**, and **Virginica**. The workflow includes data preprocessing, visualization, model training, hyperparameter tuning, and evaluation using the Random Forest Classifier.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Information](#dataset-information)
3. [Installation](#installation)
4. [Project Workflow](#project-workflow)
5. [Model Details](#model-details)
6. [Results](#results)
7. [Visualization](#visualization)
8. [Feature Importance](#feature-importance)
9. [Acknowledgments](#acknowledgments)

---

## Introduction

The Iris dataset is a classic dataset in machine learning. It contains measurements of sepal length, sepal width, petal length, and petal width for three species of iris flowers. The goal is to build a classifier that predicts the species based on these features.

---

## Dataset Information

* **Features**:

  * Sepal Length (cm)
  * Sepal Width (cm)
  * Petal Length (cm)
  * Petal Width (cm)

* **Target**:

  * Species: `setosa`, `versicolor`, `virginica`

* **Dataset Source**:

  * The dataset is included in the `sklearn.datasets` module.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/iris-classification.git
   cd iris-classification
   ```

2. Install required libraries:

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```

---

## Project Workflow

1. **Data Loading**:

   * The dataset is loaded using `sklearn.datasets.load_iris()` and converted into a Pandas DataFrame.

2. **Data Preprocessing**:

   * Checked for missing values.
   * Encoded target labels using `LabelEncoder`.
   * Scaled features using `StandardScaler`.

3. **Exploratory Data Analysis (EDA)**:

   * Used `sns.pairplot` to visualize relationships between features.
   * Observed that petal length and petal width are highly discriminative.

4. **Model Training**:

   * Split data into training (80%) and testing (20%) sets.
   * Used `RandomForestClassifier` with hyperparameter tuning via `GridSearchCV`.

5. **Evaluation**:

   * Evaluated accuracy, classification report, and confusion matrix.
   * Plotted feature importance.

---

## Model Details

* **Classifier**: Random Forest

* **Hyperparameters Tuned**:

  * `n_estimators`: \[50, 100, 150]
  * `max_depth`: \[None, 10, 20]
  * `min_samples_split`: \[2, 5, 10]

* **Best Parameters**:

  ```json
  {
    "n_estimators": 150,
    "max_depth": None,
    "min_samples_split": 2
  }
  ```

---

## Results

* **Accuracy**: 100%
* **Classification Report**:

  ```
                precision    recall  f1-score   support

            0       1.00      1.00      1.00        10
            1       1.00      1.00      1.00         9
            2       1.00      1.00      1.00        11

     accuracy                           1.00        30
    macro avg       1.00      1.00      1.00        30
  ```

weighted avg       1.00      1.00      1.00        30

```

- **Confusion Matrix**:
![Confusion Matrix](confusion_matrix.png)

---

## Visualization

1. **Pair Plot**:
 - Shows relationships between features across species.

2. **Confusion Matrix**:
 - Displays the classification results.

---

## Feature Importance

The Random Forest model highlights that **petal length** and **petal width** are the most important features for classification.

![Feature Importance](feature_importance.png)

---

## Acknowledgments

This project is based on the Iris dataset, a benchmark dataset in the field of machine learning, originally introduced by Ronald Fisher.

Feel free to contribute or suggest improvements! 😊
```
