# House Price Prediction using Machine Learning


## Project Overview
This project aims to predict residential house prices using basic structural and qualitative features from the Ames Housing dataset. Several supervised machine learning regression models are implemented and compared in terms of predictive performance, stability, and interpretability.

The project follows a fully reproducible and modular data science pipeline, from data preprocessing to model evaluation.


## Dataset
The project is based on the Ames Housing dataset, originally compiled by De Cock (2011) and distributed via the Kaggle competition *House Prices – Advanced Regression Techniques*.  
The dataset contains 1,460 observations with 79 numerical and categorical features describing residential properties in Ames, Iowa.


## Models
The following regression models are implemented and compared:
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

Regularized linear models and ensemble tree-based models are evaluated using both a validation set and cross-validation.


## Project Structure
text
├── main.py                  # Entry point
├── src/                      # Source code
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── models.py             # Model training and tuning
│   └── evaluation.py         # Evaluation and metrics
├── data/                     # Dataset files
├── notebooks/                # Exploratory analysis and figures
├── results/                  # Outputs and figures
├── requirements.txt          # Project dependencies
└── README.md


## Installation
To install the required dependencies, run the following command from the project root:

```bash
python -m pip install -r requirements.txt


## Usage
Run the complete machine learning pipeline using:

```bash
python main.py

This script loads the data, preprocesses the features, trains all models, and reports their performance using R², RMSE, and MAE.


## Reproducibility
All experiments are fully reproducible. Fixed random seeds are used throughout the pipeline to ensure consistent results across runs and environments.