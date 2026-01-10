# House Price Prediction Using Supervised Machine Learning

## Project Overview
This project aims to predict residential house prices using structural and qualitative features from the Ames Housing dataset. Several supervised machine learning regression models are implemented and compared in terms of predictive performance, stability, and interpretability.

The project follows a reproducible and modular data science pipeline, from data loading to model evaluation.

## Dataset
The project is based on the Ames Housing dataset compiled by De Cock (2011) and distributed via the Kaggle competition *House Prices – Advanced Regression Techniques*.  
The dataset contains 1,460 observations with 79 numerical and categorical features describing residential properties in Ames, Iowa.  
The data files are located in the `data/` directory.

## Models
The following regression models are implemented and compared:
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

Models are evaluated using both a validation set and 5-fold cross-validation.

## Project Structure
```text
├── main.py                   # Entry point
├── src/                      # Source code
│   ├── data_loader.py        # Data loading and train/validation split
│   ├── preprocessing.py      # Feature preprocessing
│   ├── models.py             # Model training and tuning
│   └── evaluation.py         # Evaluation and metrics
├── data/                     # Dataset files
├── notebooks/                # Exploratory analysis and figures
├── results/                  # Outputs and figures
├── requirements.txt          # Project dependencies
└── README.md
```

## Installation
Install the required dependencies from the project root using:

```bash
python -m pip install -r requirements.txt
```
**Recommended Python version:** Python 3.13.5  
The project is compatible with Python 3.9 and above. Python 3.14 is not recommended.


## Usage
Run the complete machine learning pipeline from the project root using:

```bash
python main.py
```
This script loads the data, preprocesses the features, trains all models, and reports their performance using R², RMSE, and MAE.  
No prior execution of notebooks is required.

## Reproducibility
All experiments are fully reproducible. Fixed random seeds are used throughout the pipeline to ensure consistent results across runs.