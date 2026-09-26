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
