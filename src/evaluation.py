import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


def evaluate_model(model, X_val, y_val_log):
    """Rate a model trained on log1p(SalePrice) and return the metrics in dollars."""
    y_pred_log = model.predict(X_val)

    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_val_log)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {"r2": r2, "rmse": rmse, "mae": mae}

def _rmse_dollars(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def _mae_dollars(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return mean_absolute_error(y_true, y_pred)

def cross_validate_model(model, X_train, y_train, cv=5):
    """Perform cross-validation and return mean & std of R², RMSE, and MAE."""
    scoring = {
    "r2": "r2",
    "rmse_$": make_scorer(_rmse_dollars, greater_is_better=False),
    "mae_$": make_scorer(_mae_dollars, greater_is_better=False),
    }

    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    # Compute mean and std for each metric (note: invert sign for MSE/MAE)
    r2_scores = scores["test_r2"]
    rmse_scores = -scores["test_rmse_$"]
    mae_scores = -scores["test_mae_$"]

    return {
        "r2_mean": np.mean(r2_scores),   "r2_std": np.std(r2_scores),
        "rmse_mean": np.mean(rmse_scores), "rmse_std": np.std(rmse_scores),
        "mae_mean": np.mean(mae_scores),   "mae_std": np.std(mae_scores)
    }

def get_feature_importances(model, n_top=10):
    """Get top n feature importances for tree-based or linear models."""
    # If model is a pipeline, get the underlying estimator and feature names
    feature_names = None
    final_model = model
    if hasattr(model, "named_steps"):
        # Pipeline: extract the last step (actual model) and get feature names from preprocessor
        final_model = model.named_steps["model"]
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        except AttributeError:
            feature_names = None
    # Obtain importances or coefficients if available
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        importances = final_model.coef_
    else:
        # Model does not support importance (e.g., KNN), return None
        return None
    importances = np.array(importances)
    # If feature names are known and length matches, pair them with importances
    if feature_names is not None and len(feature_names) == len(importances):
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)
        return feat_imp[:n_top]
    else:
        # Return sorted indices and values if no feature names
        indices = np.argsort(-np.abs(importances))  # indices of sorted importances (desc by abs value)
        top_indices = indices[:n_top]
        return list(zip(top_indices, importances[top_indices]))