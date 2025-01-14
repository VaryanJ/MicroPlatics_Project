# MicroPlatics_Project
# Microplastic Concentration Prediction Project

## Overview

This project focuses on predicting microplastic (MP) concentrations in various cities using machine learning models. The dataset used includes a variety of environmental and demographic features. The project employs different regression and classification models to analyze and predict MP concentrations, categorized into levels from "very low" to "very high".

## Models Implemented

The project implements and evaluates the following machine learning models:

1.  **Ordinal Logistic Regression** - Didn't work well due to failure in proportional odds assumption
2.  **Random Forest**
3.  **Support Vector Machine (SVM)**
4.  **Boosting (XGBoost)**

## Dataset

The dataset is loaded from an Excel file (`test (2).xlsx`) and includes the following features:

-   City (METRO)
-   Country
-   population
-   city area
-   population density
-   literacy rates
-   nitrogen fert. use
-   average yearly precipitation\*
-   number of vehicles
-   mismanaged plastic waste per capita
-   tidal range
-   city elevation difference (meters)
-   cars per cap
-   MP concentration (target variable)

The "MP concentration" column is the target variable and is ordinally encoded for the models as follows:

-   "very low": 0
-   "low": 1
-   "medium": 2
-   "high": 3
-   "very high": 4

## Model-Specific Details

### Ordinal Logistic Regression

-   Basic implementation without detailed feature engineering or hyperparameter tuning.

### Random Forest

-   **Libraries Used**: `pandas`, `sklearn.model_selection`, `sklearn.ensemble`, `sklearn.metrics`, `sklearn.inspection`, `matplotlib.pyplot`, `seaborn`.
-   **Feature Engineering**: Created new features such as `total_plastic_waste` and `runoff`.
-   **Cross-Validation**: Utilized LeaveOneOut cross-validation and GridSearchCV for hyperparameter tuning.
-   **Evaluation Metrics**: Confusion Matrix, Classification Report, Accuracy Score, ROC-AUC Score.
-   **Feature Importance**: Calculated and displayed using `feature_importances_`.

### Support Vector Machine (SVM)

-   **Libraries Used**: Similar to Random Forest, with the addition of `sklearn.svm` and `sklearn.preprocessing`.
-   **Feature Scaling**: Implemented `StandardScaler` for feature scaling.
-   **Hyperparameter Tuning**: Used `GridSearchCV` to find the best hyperparameters.
-   **Evaluation Metrics**: Accuracy, Classification Report, Confusion Matrix, Training and Test set scores, Matthews Correlation Coefficient (MCC), and attempted ROC-AUC Score.
-   **Feature Importance**: Permutation importance is used instead of `feature_importances_` for SVM.

### Boosting (XGBoost)

-   **Libraries Used**: `pandas`, `numpy`, `sklearn.model_selection`, `sklearn.metrics`, `sklearn.preprocessing`, `xgboost`.
-   **Cross-Validation**: Employed LeaveOneOut cross-validation.
-   **Model Training**: Used `xgb.XGBClassifier` with specified parameters for multi-class classification.
-   **Evaluation Metrics**: Confusion Matrix, Classification Report, Accuracy.

## Usage

To replicate the analysis:

1.  Ensure you have all the required libraries installed. You can install them using pip:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
    ```
2.  Place the dataset file `test (2).xlsx` in the appropriate directory.
3.  Run each cell in the notebook sequentially to execute the code.

## Results

Each model's performance is evaluated using various metrics, including accuracy, precision, recall, F1-score, and confusion matrices. Feature importance is also analyzed to understand the contribution of each variable to the model predictions.

## Notes

-   The ROC-AUC score calculation for SVM is commented out due to an error, which can be addressed for future improvements.
-   The project provides a comprehensive comparison of different machine learning models for predicting microplastic concentrations based on environmental and demographic data.
-   The specific hyperparameters used in each model and the detailed results are printed within the notebook execution outputs.
