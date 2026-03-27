# House Price Prediction: Linear Models

The project covers the full ML lifecycle: EDA, feature engineering with statistical validation, regularized regression, and inference on unseen data.

## Methodology

1. **EDA** — distribution analysis, neighborhood price comparison, outlier detection
2. **Feature Engineering** — multicollinearity analysis (VIF), partial correlation, feature creation (`HouseAge`)
3. **Preprocessing Pipeline** — `ColumnTransformer` with separate strategies for numeric and categorical features
4. **Model Training** — Linear Regression (baseline), Ridge, Lasso with cross-validated alpha selection
5. **Inference** — predictions on unseen test data with distribution comparison and outlier analysis

## Feature Engineering

This was the most analytical part of the project. Three statistical tools were used to make principled decisions:

- **VIF (Variance Inflation Factor)** — detected multicollinearity across all features simultaneously, not just pairwise. Identified perfect multicollinearity in composite features (`TotalBsmtSF`, `GrLivArea`) and removed their components.
- **Partial Correlation** — used to decide which feature to keep from each correlated pair. Measures the *unique* contribution of each feature to `SalePrice` after removing shared variance with the competing feature.
- **Pearson Correlation** — confirmed the direction of linear relationships and guided feature selection.

| Dropped Feature | Kept | Partial r (dropped) | Partial r (kept) |
|----------------|------|---------------------|-----------------|
| TotRmsAbvGrd | GrLivArea | -0.19 | 0.61 |
| GarageYrBlt | YearBuilt | 0.13 | 0.26 |
| GarageArea | GarageCars | 0.17 | 0.23 |

`YearBuilt` was replaced with `HouseAge = YrSold - YearBuilt` for improved interpretability — each additional year of age is associated with a lower sale price, which is more intuitive than a raw construction year.

## Results

| Model | Train RMSE | Validation RMSE | Gap |
|-------|-----------|-----------------|-----|
| Linear Regression | 0.0879 | 0.1283 | 0.041 |
| Ridge (α=19.47) | 0.1010 | 0.1171 | 0.016 |
| **Lasso (α=0.00066)** | **0.1040** | **0.1160** | **0.012** |

RMSE is measured in log(SalePrice) units — a validation RMSE of 0.1160 corresponds to a median prediction error of 6.8%.

**Lasso was selected** as the best model:
- lowest validation RMSE
- smallest overfitting gap
- automatic feature selection: 162 of 245 features (66%) were zeroed out

<img width="1999" height="948" alt="image" src="https://github.com/user-attachments/assets/cf9bbee1-7408-4b0d-a392-b100be92e0f7" />

Lasso predictions closely follow actual prices across the full validation set (292 homes). Larger deviations are visible at the extremes of the price distribution.

<img width="1744" height="929" alt="image" src="https://github.com/user-attachments/assets/0f0ffa6e-532d-4f7a-a511-1317e5871745" />


## Key Findings

- Living area (`GrLivArea`) and overall quality (`OverallQual`) are the strongest positive predictors of sale price
- Neighborhood has a significant impact: premium neighborhoods (`Neighborhood_StoneBr`, `Neighborhood_NridgHt`, `Neighborhood_Crawfor`) are associated with substantially higher prices
- New construction (`SaleType_New`) tends to sell for higher prices
- `HouseAge` has the strongest negative effect: each additional year of age is associated with a lower sale price
- `Neighborhood_MeadowV` and townhouse building type (`BldgType_Twnhs`) are associated with lower prices compared to the reference categories

## Limitations

- The model is unreliable for homes larger than those in the training set (max: 4,476 sqft) — the largest test set home (5,095 sqft) received a predicted price of $1.5M vs ~$750k for comparable training homes
- A non-linear relationship between `HouseAge` and `SalePrice` was identified during feature engineering. A degree-3 polynomial fit captures the pattern significantly better than a linear fit: prices drop steeply for homes 
aged 15–60 years, then stabilize. Linear regression cannot fully capture this non-linear relationship, leading to systematic errors for very new and very old homes.

<img width="2188" height="620" alt="image" src="https://github.com/user-attachments/assets/f93d78d2-1533-4cdc-9abc-dc21ce73de5f" />

## Next Steps

A non-linear relationship between `HouseAge` and `SalePrice` was identified during feature engineering and confirmed visually using polynomial regression fits (degree 1–3).

A follow-up project will focus on nonlinear modeling and comparing Polynomial Regression, Step Functions, and Regression Splines with the goal of incorporating the best method into this pipeline.

## Libraries

- `pandas`, `numpy` — data manipulation
- `scikit-learn` — pipeline, preprocessing, model training
- `statsmodels`, `pingouin` — VIF, partial correlation
- `seaborn`, `matplotlib` — visualization
- `joblib` — saving and loading pipeline components
