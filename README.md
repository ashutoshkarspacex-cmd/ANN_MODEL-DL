# 🧠 Optimized ANN Classifier: RFE & Bayesian Optimization

This repository contains a machine learning pipeline for building, optimizing, and evaluating an Artificial Neural Network (Multi-Layer Perceptron) using TensorFlow/Keras. 

This project goes beyond a standard neural network implementation by focusing heavily on **dimensionality reduction** and **hyperparameter tuning**. It utilizes Recursive Feature Elimination (RFE) to strip away noisy data and Optuna (Bayesian Optimization) to dynamically discover the optimal network architecture.

## 🚀 Key Features

* **Advanced Feature Selection:** Uses Scikit-Learn's `RFE` with a `RandomForestClassifier` estimator to reduce the dataset down to the top 10 most critical features, preventing the curse of dimensionality and reducing overfitting.
* **Bayesian Hyperparameter Optimization:** Implements `Optuna` to intelligently search for the best model architecture in a randomforestclassifier(number of trees and maximum depth of tree) using a Tree-structured Parzen Estimator (TPE), vastly outperforming standard Grid/Random search.
* **Deep Learning Architecture:** A dynamic Multi-Layer Perceptron (MLP) built with TensorFlow/Keras, utilizing Early Stopping to prevent overfitting during training.
* **Robust Preprocessing Pipeline:** Includes strict automated handling of missing values (`NaN` imputation/dropping) and feature standardization (`StandardScaler`) to ensure gradient descent converges smoothly.
* **Comprehensive Evaluation:** Evaluates model performance using Accuracy alongside Confusion Matrix to account for potential class imbalances.

## 📁 Repository Contents

* `ANN_MODEL_WITH_OPTIMISATION.ipynb`: The core Jupyter Notebook containing the end-to-end pipeline. It covers data loading, sanitization, scaling, feature selection, Optuna optimization, model training, and evaluation.
* `rfe.pickle`: A serialized (pickled) instance of the fitted Scikit-Learn Recursive Feature Elimination object. This allows for immediate transformation of new, unseen data without needing to retrain the Random Forest selector from scratch.

## 🛠️ Tech Stack & Requirements

To run this project locally, you will need Python 3.8+ and the following libraries:
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn optuna


## 🏗️ Pipeline Architecture & Workflow

1. **Data Sanitization:** Strict dropping or imputation of `NaN` values to prevent gradient explosion and constant-prediction states (a common pitfall in local Pandas environments).
2. **Feature Scaling:** All features are passed through a `StandardScaler` ($\mu=0$, $\sigma^2=1$) to stabilize the Hessian matrix curvature and ensure stable weight updates during backpropagation.
3. **Feature Selection (RFE):** The `rfe.pickle` model identifies and filters the dataset down to the optimal subset of features based on node impurity decreases.
4. **Architecture Search (Optuna):** An objective function dynamically builds and evaluates sequential models, tweaking layers and learning rates to minimize validation loss.
5. **Final Training & Evaluation:** The best architecture is trained on the RFE-selected data and evaluated across multiple metrics.

## 📊 Results

* **Feature Reduction:** The RFE algorithm successfully isolated the top 10 most impactful features, filtering out mathematical noise and reducing dimensionality.
* **Hyperparameter Tuning:** Optuna successfully identified the optimal learning rate and hidden layer density through Bayesian search.
* **Performance Metrics:** The fully optimized ANN achieved a highly stable test accuracy of **~79.2%**, with detailed breakdowns available via the classification report and confusion matrix within the notebook.
