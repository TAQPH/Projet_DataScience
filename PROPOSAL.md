# Project Proposal – Predicting Real Estate Prices 

## Project Category 
Data Analysis & Visualization / Supervised Machine Learning 


## 1 Project Title 
Predicting Housing Prices 


## 2 Problem Statement 
Understanding what drives housing prices is an important topic for both policymakers 
and investors. Real estate prices depend on many factors such as location, size, 
building quality, and local economic conditions. Traditional models use linear regression 
to explain prices, but the relationships between property characteristics and market 
value can be highly non-linear.   
The goal of this project is to build and compare machine learning models to predict 
housing prices from property-level features. By doing so, the project will highlight which 
factors contribute most to house values and evaluate how advanced methods improve 
prediction accuracy over simple regression models. 


## 3 Data and Sources 
The analysis will use the “House Prices – Advanced Regression Techniques” dataset 
from Kaggle.   
The dataset contains information on around 1460 houses in Ames, Iowa, including 80 
variables describing physical attributes (ex : area, number of rooms, garage size, …) and 
sale price.   
Additional cleaning and preprocessing will be performed in Python (handling missing 
values, encoding categorical variables, and normalizing numeric features). 


## 4 Planned Approach and Technologies 
Python environment with the following libraries : - - - 
‘pandas’ and ‘numpy’ for data manipulation,   
‘matplotlib’ and ‘seaborn’ for visualization,   
‘scikit-learn’ for model training and evaluation.   

### Steps: 

1. Data preprocessing   - - - - - - - - - - - - - 
Handle missing values and outliers.   
Encode categorical variables using one-hot encoding.   
Normalize or scale numeric features where needed. 

2. Exploratory analysis   
Visualize distributions and correlations between main variables (e.g., living area vs. 
price).   
Identify potential non-linear relationships. 

3. Model building   
Baseline: Linear Regression (OLS).   
Regularized models: Ridge and Lasso regression.   
Advanced models: Random Forest Regressor and Gradient Boosting Regressor.   
Each model will be tuned using grid search and cross-validation. 

4. Model evaluation   
Evaluate predictive performance using R², Mean Squared Error (MSE), and Mean 
Absolute Error (MAE).   
Compare results across models and discuss trade-offs between interpretability 
and accuracy. 

5. Interpretation   
Analyze feature importance from tree-based models to identify the main price 
determinants.   
Discuss economic intuition (e.g., larger area and better overall quality increase 
value). 


## 5 Expected Challenges and Mitigation - - -

High dimensionality : use feature selection and regularization (Ridge/Lasso) to 
avoid overfitting.   
Missing or categorical data : apply proper preprocessing and encoding.   
Model complexity : compare interpretable (linear) and black-box (ensemble) 
approaches. 


## 6 Success Criteria - - - - 

Clean and reproducible code implementing at least three regression models.   
Sound comparison showing the performance gain from machine learning 
methods.   
Graphical and quantitative interpretation of the most relevant features.   
A concise report explaining data, methods, results, and limitations. 


## 7) Stretch Goals (If Time Permits) - - - 

Add XGBoost or LightGBM for improved gradient boosting performance.   
Use SHAP values for advanced model interpretability.   
Test model robustness on another housing dataset (ex : California housing data).