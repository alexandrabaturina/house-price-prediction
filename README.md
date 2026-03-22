# House Price Prediction

Predicting residential sale prices using the structured house data. 

The project covers the full ML lifecycle: EDA, feature engineering with statistical validation, regularized regression, and inference on unseen data.

## Dataset

The dataset contains 1,460 training records with 79 features describing residential properties. The target variable is `SalePrice`.

## Project Structure
```
house-price-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
├── house_price_prediction.ipynb
├── .gitignore
└── README.md
```

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
| Linear Regression | 0.0882 | 0.1283 | 0.0401 |
| Ridge (α=18) | 0.1011 | 0.1172 | 0.0161 |
| **Lasso (α=0.00066)** | **0.1042** | **0.1162** | **0.0120** |

RMSE is measured in log(SalePrice) units — a validation RMSE of 0.1162 corresponds to a median prediction error of 6.6%.

**Lasso was selected** as the best model:
- lowest validation RMSE
- smallest overfitting gap
- automatic feature selection: 161 of 245 features (66%) were zeroed out

<img width="2030" height="974" alt="image" src="https://github.com/user-attachments/assets/2cfe2622-0fe8-4aaa-bd30-f34de89b842c" />

Lasso predictions closely follow actual prices across the full validation set (292 homes). Larger deviations are visible at the extremes of the price distribution.

<img width="1733" height="933" alt="image" src="https://github.com/user-attachments/assets/d5543832-aa2d-4b3a-9b30-5bfbee430e77" />


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

<img width="2201" height="612" alt="image" src="https://github.com/user-attachments/assets/eedf9813-527a-4184-95bf-886083eddb17" />

## Next Steps

A non-linear relationship between `HouseAge` and `SalePrice` was identified during feature engineering and confirmed visually using polynomial regression fits (degree 1–3). A follow-up project will focus on comparing Polynomial Regression, Step Functions, and Regression Splines with the goal of incorporating the best method into this pipeline.

## Libraries

- `pandas`, `numpy` — data manipulation
- `scikit-learn` — pipeline, preprocessing, model training
- `statsmodels`, `pingouin` — VIF, partial correlation
- `seaborn`, `matplotlib` — visualization
- `joblib` — saving and loading pipeline components
