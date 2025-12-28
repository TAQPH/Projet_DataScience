from src import data_loader, models, evaluation

# 1. Data loading
X_train, X_val, y_train, y_val, X_test = data_loader.load_data(log_target=True)

# 2. Train models with their pipelines and hyperparameter tuning
linear_model = models.train_linear_model(X_train, y_train)
ridge_model = models.train_ridge_model(X_train, y_train)
lasso_model = models.train_lasso_model(X_train, y_train)
rf_model = models.train_rf_model(X_train, y_train)
gb_model = models.train_gb_model(X_train, y_train)

# 3. Evaluate each model on validation set and with cross-validation
models_list = [("Linear Regression", linear_model),
               ("Ridge Regression", ridge_model),
               ("Lasso Regression", lasso_model),
               ("Random Forest", rf_model),
               ("Gradient Boosting", gb_model)]

print("Validation Set Performance:")
for name, model in models_list:
    scores = evaluation.evaluate_model(model, X_val, y_val)
    print(f"{name}: R² = {scores['r2']:.3f}, RMSE = {scores['rmse']:.2f}, MAE = {scores['mae']:.2f}")

print("\nCross-Validation Performance (5-fold):")
for name, model in models_list:
    cv_scores = evaluation.cross_validate_model(model, X_train, y_train, cv=5)
    print(f"{name}: R²(log) = {cv_scores['r2_mean']:.3f} (±{cv_scores['r2_std']:.3f}), "
          f"RMSE = {cv_scores['rmse_mean']:.2f} (±{cv_scores['rmse_std']:.2f}), "
          f"MAE = {cv_scores['mae_mean']:.2f} (±{cv_scores['mae_std']:.2f})")

# 4. Display top feature importances for tree-based models (and linear coefficients)
print("\nTop Features by Importance:")
for name, model in models_list:
    importances = evaluation.get_feature_importances(model, n_top=10)
    if importances is not None:
        print(f"{name} - Top {len(importances)} features:")
        for feat, imp in importances:
            print(f"    {feat}: {imp:.4f}")
        print()