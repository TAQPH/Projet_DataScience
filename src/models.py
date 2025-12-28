import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(X: pd.DataFrame):
    """Construct a preprocessing pipeline for numeric and categorical features."""
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    # Numeric pipeline: impute missing values then scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: impute missing (as new category) then one-hot encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines for numeric and categorical data
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def train_linear_model(X_train, y_train):
    """Train a Linear Regression model (no hyperparameters to tune)."""
    preprocessor = build_preprocessor(X_train)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    param_grid = {}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error")
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_ridge_model(X_train, y_train):
    """Train a Ridge Regression model with hyperparameter tuning on alpha."""
    preprocessor = build_preprocessor(X_train)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ])

    param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error")
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_lasso_model(X_train, y_train):
    """Train a Lasso Regression model with hyperparameter tuning on alpha."""
    preprocessor = build_preprocessor(X_train)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Lasso(max_iter=50000))
    ])

    param_grid = {"model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error")
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_rf_model(X_train, y_train):
    """Train a Random Forest model with hyperparameter tuning."""
    preprocessor = build_preprocessor(X_train)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__max_features": ["sqrt", "log2", None],
    }

    grid = GridSearchCV(
        pipe, param_grid, cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_gb_model(X_train, y_train):
    """Train a Gradient Boosting model with hyperparameter tuning."""
    preprocessor = build_preprocessor(X_train)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.1, 0.01],
        "model__max_depth": [3, 5]
    }

    grid = GridSearchCV(
        pipe, param_grid, cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_