# Movie Revenue Prediction Using Machine Learning 

## Project Overview:
This project develops a supervised regression system for predicting movie revenue from structured movie attributes. The supplied dataset contains 4,760 records and 21 original fields. The workflow covers problem definition, exploratory analysis, preprocessing, feature engineering, model comparison, evaluation and deployment planning. Revenue is modeled as log(1 + revenue) because of strong right-skewness. Numerical predictors are median-imputed and standardized; categorical predictors are most-frequent-imputed and one-hot encoded. Linear Regression, Ridge Regression and Random Forest Regression are compared using a fixed 80/20 split. The final model is selected from measured test-set results and saved as a joblib pipeline.

### Problem Statement
Given pre-release or release-associated movie attributes, estimate Movie_Revenue using supervised regression.

### Project Objectives
•	Understand data quality and distributions.

•	Perform at least eight EDA visualizations.

•	Handle missing values and duplicates and address skewness.

•	Engineer release year and month.

•	Compare multiple regression models.

•	Evaluate using MAE, RMSE and R².

•	Discuss generalization, overfitting and bias-variance.

•	Prepare a deployment-ready model artifact and API design.

### Success Criteria
The solution should be reproducible, use leakage-safe preprocessing, report actual test-set metrics, compare multiple models, and produce a deployable serialized pipeline.

### Dataset Information
Dataset Source: https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv

### Dataset Description
The supplied Movies Recommendation.csv file contains 4,760 records and 21 original fields, exceeding the dataset-size requirements in the supplied guidelines.

<img width="524" height="147" alt="image" src="https://github.com/user-attachments/assets/47e8196c-e1c0-4fa0-9f24-d18f5abc54fe" />

### Why is this a supervised learning project?

Because the training dataset already contains the correct answer:

Movie attributes → Actual Revenue

The model learns from these known answers.

                    SUPERVISED LEARNING

         INPUT FEATURES              TARGET
       ┌─────────────────┐        ┌───────────────┐
       │ Budget          │        │               │
       │ Popularity      │        │               │
       │ Runtime         │───────►│    Revenue    │
       │ Vote            │        │               │
       │ Genre           │        │               │
       │ Language        │        │               │
       │ etc.            │        │               │
       └─────────────────┘        └───────────────┘
              X                         Y
### Project Structure
MOVIE REVENUE PREDICTION


├── 1. Problem Definition

├── 2. Dataset Understanding

├── 3. Data Cleaning

├── 4. Exploratory Data Analysis

├── 5. Data Preprocessing

├── 6. Feature Engineering

├── 7. Feature Selection

├── 8. Train-Test Split

├── 9. Linear Regression

├── 10. Ridge Regression

├── 11. Random Forest Regression

├── 12. Model Evaluation

├── 13. Model Comparison

├── 14. Prediction

├── 15. Conclusion

└── 16. Deployment

### PROJECT WORK FLOW:

                INPUT
                  ↓
       ┌─────────────────────┐
       │ Movie Budget        │
       │ Movie Popularity    │
       │ Movie Runtime       │
       │ Movie Vote          │
       │ Movie Vote Count    │
       │ Release Year        │
       │ Release Month       │
       │ Movie Genre         │
       │ Movie Language      │
       │ Production Country  │
       └─────────────────────┘
                  ↓
           PREPROCESSING
                  ↓
        RANDOM FOREST MODEL
                  ↓
                OUTPUT
                  ↓
        Predicted Movie Revenue

### Project Workflow Summary

The complete procedure can be summarized as:

Step 1 – Define Problem

Predict movie revenue using supervised machine learning.

Step 2 – Load Dataset

Load Movies Recommendation.csv.

Step 3 – Understand Dataset

Analyze rows, columns, data types and statistics.

Step 4 – Clean Data

Handle missing values and remove duplicate records.

Step 5 – Perform EDA

Analyze revenue, budget, popularity, ratings and other variables using statistical summaries and visualizations.

Step 6 – Engineer Features

Extract release year and release month from the release date.

Step 7 – Select Features

Remove identifiers and unsuitable variables.

Step 8 – Transform Target

Apply:

np.log1p()

to revenue.

Step 9 – Encode Categories

Apply One-Hot Encoding.

Step 10 – Scale Numerical Features

Apply StandardScaler.

Step 11 – Split Dataset

Use 80% training and 20% testing data.

Step 12 – Train Models

Train:

Linear Regression
Ridge Regression
Random Forest Regression
Step 13 – Evaluate Models

Calculate:

MAE
RMSE
R²
Step 14 – Compare Models

Compare the three models based on their test-set performance.

Step 15 – Select Final Model

Select the model based on the actual evaluation results.

Step 16 – Save Model

Save the complete preprocessing and prediction pipeline using Joblib.

Step 17 – Develop API

Create a Flask /predict endpoint.

Step 18 – Containerize

Create Docker configuration for deployment


### Results and Discussion
The executed experiment compares three regression approaches using the same feature set and preprocessing. The measured test-set results identify Random Forest Regression under the selected RMSE criterion. The result is data-derived rather than assumed.
The experiment also demonstrates the value of a unified pipeline: categorical expansion can create a high-dimensional feature matrix, numerical scaling affects linear models, and non-linear ensembles provide a useful contrast to linear assumptions.

### Business Interpretation
The model is an analytical estimator, not a guaranteed financial forecast. Movie revenue can depend on marketing, distribution, franchise strength, competition, release strategy, audience trends and other factors not represented in the baseline dataset.

### Limitations and Future Enhancements
•	Validate the dataset's original source and definitions before production use.
•	Add TF-IDF or embeddings for overview/keywords/tagline.
•	Explore cast, crew and director features with careful leakage controls.
•	Use time-aware validation for future-release prediction.
•	Tune hyperparameters with cross-validation.
•	Estimate prediction intervals or uncertainty.
•	Add external business variables such as marketing or competition.
•	Monitor model drift after deployment.

### Conclusion
This project implements an end-to-end supervised learning workflow for movie revenue prediction using the supplied dataset. It covers EDA, preprocessing, feature engineering, multiple-model training, evaluation and deployment planning. The executed experiment selects Random Forest Regression under the stated test-set RMSE criterion, and the complete fitted pipeline has been saved as a joblib artifact.
The project satisfies the major technical workflow elements specified in the supplied mini-project guidelines while leaving richer text features, stronger validation and external variables as future enhancements.


### Start Streamlit
Open another terminal:

streamlit run app.py
The Streamlit application will open in your browser.

# Data Science Mini Project - Author

**Student Name   : ASWINKUMAR R**

**Batch          : DS AN B03**

**Project Title  : Movie Revenue Prediction USing Machine Learning**

**Submission Date: 26/09/26**

