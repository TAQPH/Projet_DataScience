import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(train_path="data/train.csv", test_path="data/test.csv", log_target=True):
    """Load the data, handle outliers, and split into train/validation sets."""
    # Load CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop unnecessary identifier column if present
    if "Id" in train_df.columns:
        train_df = train_df.drop(columns="Id")
    if "Id" in test_df.columns:
        test_df = test_df.drop(columns="Id")
    
    # Handle outliers in training data (e.g., remove extremely large houses with unusually low price)
    # On the Kaggle House Prices data, two outliers have GrLivArea > 4000 and SalePrice < 300000
    train_df = train_df.drop(train_df[(train_df["GrLivArea"] > 4000) & (train_df["SalePrice"] < 300000)].index)
    # Remove any house with an exceptionally large basement area as an outlier (if applicable)
    if "TotalBsmtSF" in train_df.columns:
        train_df = train_df.drop(train_df[train_df["TotalBsmtSF"] > 6000].index)

    
    # Separate features and target
    target_col = "SalePrice"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")
    
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    
    # Split into training and validation sets (e.g., 80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log-transform of the target variable
    if log_target:
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
    
    # Prepare test features (note: test set has no target in this scenario)
    X_test = test_df.copy()

    
    return X_train, X_val, y_train, y_val, X_test