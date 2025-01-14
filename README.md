# Microplastic Concentration Prediction

This project focuses on predicting microplastic (MP) concentration in aquatic environments using machine learning models. The models utilize various environmental and socioeconomic factors as predictors.

## Project Structure

The project consists of three main sections, each corresponding to a different machine learning model:

1.  **Random Forest:** This section implements a Random Forest Classifier to predict MP concentration based on selected features.
2.  **SVM (Support Vector Machine):** This section utilizes a Support Vector Classifier for MP concentration prediction, including hyperparameter tuning and feature importance analysis.
3.  **Boosting (XGBoost):** This section employs an XGBoost Classifier for prediction, featuring Leave-One-Out Cross-Validation and feature importance visualization.

## Data

The dataset `test (2).xlsx` is used for this project. It contains information on various factors such as population density, literacy rates, nitrogen fertilizer use, average yearly precipitation, mismanaged plastic waste per capita, and tidal range. The target variable is "MP concentration," which is ordinally encoded into five categories:

-   very low: 0
-   low: 1
-   medium: 2
-   high: 3
-   very high: 4
## Dataset

The dataset is loaded from an Excel file (`test (2).xlsx`) named (`CityBasedDataSet.xlsx`) in this GitHub and includes the following features:

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

Database: https://experience.arcgis.com/experience/b296879cc1984fda833a8acc93e31476
Data was manually gathered from the above dataset looking at data that was within 24 miles of the city, thus within international waters. 

## Dependencies

The following Python libraries are required to run the code:

-   pandas
-   scikit-learn
-   matplotlib
-   seaborn
-   xgboost

## Code Description

### Random Forest

1.  **Data Loading and Preprocessing:**
    -   Loads the dataset from `test (2).xlsx`.
    -   Removes the first two columns (indexed 0 and 1).
    -   Ordinally encodes the "MP concentration" column into "MP\_Concentration\_Encoded" using a predefined mapping.
    -   Creates new features: 'total\_plastic\_waste' and 'runoff'.

2.  **Model Training and Evaluation:**
    -   Sets 'population density', 'literacy rates', 'nitrogen fert. use', 'number of vehicles', 'mismanaged plastic waste per capita', and 'tidal range' as features (X) and "MP\_Concentration\_Encoded" as the target variable (y).
    -   Splits the data into training and testing sets (80/20 split).
    -   Initializes LeaveOneOut cross-validation.
    -   Defines a parameter grid for hyperparameter tuning using `GridSearchCV`.
    -   Initializes a `RandomForestClassifier` with `random_state=42`.
    -   Performs `GridSearchCV` with 4-fold cross-validation to find the best hyperparameters.
    -   Trains the best model on the entire training set.
    -   Performs Leave-One-Out Cross-Validation (LOOCV) for a more robust evaluation on the training set, storing predictions and true labels.
    -   Evaluates the model using the LOOCV predictions and true labels, printing the confusion matrix, classification report, and accuracy.
    -   Calculates and prints the ROC-AUC score on the test set.
    -   Evaluates the model on the test set, printing the confusion matrix and accuracy.
    -   Calculates and visualizes feature importance using a bar plot.

### SVM

1.  **Data Loading and Preprocessing:**
    -   Loads the dataset from `test (2).xlsx`.
    -   Removes the first two columns (indexed 0 and 1).
    -   Strips leading/trailing whitespace from the "MP concentration" column.

2.  **Model Training and Evaluation:**
    -   Sets all columns except the last as features (X) and "MP concentration" as the target variable (y).
    -   Splits the data into training and testing sets (80/20 split).
    -   Scales the features using `StandardScaler`, fitting on the training data and transforming both training and testing data.
    -   Defines a parameter grid for hyperparameter tuning using `GridSearchCV`.
    -   Trains and tunes an `SVC` model with `class_weight='balanced'` using `GridSearchCV` with 4-fold cross-validation.
    -   Evaluates the best model on the test set, printing the best hyperparameters, accuracy, classification report, and confusion matrix.
    -   Calculates and prints the Matthews Correlation Coefficient (MCC).
    -   Calculates and visualizes feature importance using permutation importance.

### Boosting

1.  **Data Loading and Preprocessing:**
    -   Loads the dataset from `test (2).xlsx`.
    -   Removes the first two columns (indexed 0 and 1).
    -   Ordinally encodes the "MP concentration" column into "MP\_Concentration\_Encoded" using a predefined mapping.

2.  **Model Training and Evaluation:**
    -   Sets 'population density', 'literacy rates', 'nitrogen fert. use', 'average yearly precipitation\*', 'mismanaged plastic waste per capita', and 'tidal range' as features (X) and "MP\_Concentration\_Encoded" as the target variable (y).
    -   Initializes LeaveOneOut cross-validation.
    -   Defines an `XGBClassifier` with specified hyperparameters.
    -   Performs LOOCV, standardizing the features within each fold using `StandardScaler`, training the model on the training fold, and making predictions on the test fold.
    -   Evaluates the model using the predictions and true labels from all folds, printing the confusion matrix, classification report, and accuracy.
    -   Calculates and visualizes feature importance using a bar plot.

## Usage

1.  Ensure you have the required libraries installed. You can install them using pip:

    ```bash
    pip install pandas scikit-learn matplotlib seaborn xgboost
    ```

2.  Download the dataset `test (2).xlsx` and place it in the same directory as the script.
3.  Run the Jupyter Notebook or Python script containing the code.

## Results

The code outputs the following for each model:

-   **Training Set (LOOCV for Random Forest and XGBoost):**
    -   Confusion Matrix
    -   Classification Report
    -   Accuracy Score
-   **Test Set:**
    -   Confusion Matrix
    -   Accuracy Score
    -   ROC-AUC Score (for Random Forest)
    -   Matthews Correlation Coefficient (MCC) (for SVM)
-   **Feature Importance:**
    -   A bar plot visualizing the importance of each feature.

## Notes

-   The code uses Leave-One-Out Cross-Validation (LOOCV) for the Random Forest and Boosting models to provide a more robust estimate of model performance, especially given the potentially small dataset size.
-   The SVM model uses `GridSearchCV` for hyperparameter tuning.
-   Feature importance is calculated using different methods for each model:
    -   Random Forest: Gini importance (built-in to the model).
    -   SVM: Permutation importance.
    -   XGBoost: Gini importance (built-in to the model).
-   The code includes links to external resources for further information on LOOCV, ROC-AUC, and other evaluation metrics.
-   The Random Forest and Boosting code includes explicit references to external resources like tutorials and documentation where certain code snippets or concepts were sourced from. This is good practice for transparency and acknowledging sources.
-   The SVM code includes an explicit reference to a GitHub Gist that was potentially used for guidance on scaling features.

This README provides a comprehensive overview of the project, enabling users to understand, run, and interpret the results of the provided code.
-   The project provides a comprehensive comparison of different machine learning models for predicting microplastic concentrations based on environmental, demographic, and economic data.
-   The specific hyperparameters used in each model and the detailed results are printed within the notebook execution outputs.

##Future Improvements and Goals
In the future, we hope to implement more parameters, gather data from more cities, and generalize our results - conducting further analysis. With the project, we can figure out what land-based parameters coorelate well with microplastic concentrations in nearby coastal water which can have large impacts in the lab and in policy. 
