# Results

## Validation set performance

**Table 1 – Validation set metrics**

| Model                |   R²   | RMSE ($) | MAE ($) |
|----------------------|:------:|---------:|--------:|
| Linear Regression    | 0.914  | 21,785   | 15,454  |
| Ridge Regression     | 0.927  | 20,036   | 14,509  |
| Lasso Regression     | 0.933  | 19,293   | 14,033  |
| Random Forest        | 0.895  | 24,116   | 16,519  |
| Gradient Boosting    | 0.925  | 20,331   | 14,892  |

Lasso achieves the best validation performance.  
Regularized linear models outperform tree-based models.

---

## Cross-validation performance

5-fold cross-validation.  
R² is computed on log-prices.  
RMSE and MAE are reported in dollars.

**Table 2 – Cross-validation metrics**

| Model                | R² (log) | RMSE ($) | MAE ($) |
|----------------------|:--------:|---------:|--------:|
| Linear Regression    | 0.889    | 23,256   | 15,178  |
| Ridge Regression     | 0.915    | 20,394   | 13,615  |
| Lasso Regression     | 0.913    | 20,573   | 13,697  |
| Random Forest        | 0.872    | 28,381   | 17,526  |
| Gradient Boosting    | 0.898    | 24,392   | 15,190  |

Ridge and Lasso are the most stable models.  
Random Forest shows higher variance.

---

## Feature importance

**Table 3 – Main predictive features**

| Category | Features |
|--------|----------|
| Quality | `OverallQual` |
| Size | `GrLivArea`, `TotalBsmtSF` |
| Garage | `GarageArea`, `GarageCars` |
| Age | `YearBuilt`, `YearRemodAdd` |
| Location | `Neighborhood_*`, `MSZoning_*` |

Important features are consistent across models.

---

## Summary

Regularized linear models perform best.  
Tree-based models do not provide clear gains.  
Ridge and Lasso are suitable final models.