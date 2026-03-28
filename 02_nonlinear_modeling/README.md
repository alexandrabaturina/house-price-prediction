# House Price Prediction: Nonlinear Modeling

This project explores nonlinear relationships between selected features and `SalePrice`, building directly on the [Linear Models project](https://github.com/alexandrabaturina/house-price-prediction/tree/main/01_linear_models) 

Three methods are compared across three features, and the best-performing transformations are integrated into the Lasso pipeline.

## Methodology
1. **Feature Selection** — three features selected based on the shape of their relationship with `SalePrice`:
   
    -  `HouseAge` (U-shaped)
    -  `TotalBsmtSF` (accelerating trend)
    -  `OverallQual` (stepwise)
3. **Nonlinear Modeling** — three different models are fitted and compared for each feature:
    - Polynomial Regression
    - Step Functions
    - Regression Splines
5. **Evaluation**
     
    -  RMSE
    -  bias-variance tradeoff
    -  interpretability
6. **Pipeline Integration** — best-performing transformations incorporated into the Lasso pipeline with re-tuned alpha

## Feature Selection

| Feature | Relationship Type | Best Method |
|---|---|---|
| HouseAge | U-shaped curve | Splines (n_knots=6) |
| TotalBsmtSF | Accelerating trend | Splines (n_knots=4) |
| OverallQual | Stepwise | Step Functions |

<img width="3085" height="872" alt="image" src="https://github.com/user-attachments/assets/22334fa4-d962-4141-848d-68bd752e25f7" />

## Results

| Model | Train RMSE | Validation RMSE | Gap |
|---|---|---|---|
| Lasso baseline (α=0.00066) | 0.1040 | 0.1160 | 0.0120 |
| Lasso + Splines (α=0.000762) | 0.1043 | 0.1150 | 0.0107 |

Applying spline transformations to `HouseAge` and `TotalBsmtSF` improved the baseline Lasso model from 0.1160 to 0.1150 validation RMSE.
> RMSE is measured in log(SalePrice) units.

## Key Findings

| Feature | Linear RMSE | Best Nonlinear RMSE | Improvement | Best Method |
|---|---|---|---|---|
| HouseAge | 0.3258 | 0.3090 | 0.0168 | Splines (n_knots=6) |
| TotalBsmtSF | 0.3123 | 0.3093 | 0.0030 | Splines (n_knots=4) |
| OverallQual | 0.2340 | 0.2332 | 0.0008 | Step Functions |

* `HouseAge` benefited most from nonlinear modeling — splines reduced validation RMSE by 0.0168 compared to a linear baseline
* `OverallQual` showed negligible improvement (0.0008) — the relationship is already nearly linear on the log scale
* Step functions are naturally suited for discrete ordinal features, while splines are appropriate for continuous features
* In a model with 83 active features, transforming two mid-importance predictors has a limited global impact

## Limitations

* Splines treat structural zeros (homes without a basement) as part of the continuous range, which affects predictions for `TotalBsmtSF`=0
* Both splines and polynomials are unreliable in sparse regions (e.g. `TotalBsmtSF` > 2000, `HouseAge` > 100)

## Next Steps
A follow-up project will use the updated Lasso pipeline as a baseline for a feedforward neural network, with performance compared across Linear Regression and deep learning.

## Libraries
* `pandas`, `numpy` — data manipulation
* `scikit-learn` — pipeline, preprocessing, model training
* `seaborn`, `matplotlib` — visualization
* `joblib`, `pickle` — saving and loading pipeline components
