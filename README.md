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
<img width="540" height="143" alt="image" src="https://github.com/user-attachments/assets/082e408f-fbd8-42c1-920f-b32a1b23da13" />
<img src="https://github.com/user-attachments/assets/082e408f-fbd8-42c1-920f-b32a1b23da13" alt="Dataset Details" width="400" height="200">



