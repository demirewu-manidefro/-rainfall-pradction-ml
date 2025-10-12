Rainfall Prediction Project 🌧️

Developed for: Ethiopian Statistics Service (ESS)
Prepared by: Demirewu Manidefro
Program: Data Science Internship
Duration: 2 Months
Supervisor: Stotaw (ESS Manager)
Date: October 2025

Overview

This project predicts rainfall using environmental, geographical, and climatic features. The goal is to create an accurate model to support planning and decision-making. Multiple models were trained, evaluated, and compared to select the best performing approach.

Key steps include:

Data preprocessing (outlier handling, log transformation, encoding, scaling)

Feature selection (multicollinearity reduction using VIF)

Model training and evaluation (Linear Regression, Decision Tree, Random Forest)

Dataset

Size: 514 observations, 33 raw features + 2 categorical variables (ssa_aez09, landcov)

Features:

Distances: dist_road, dist_market, dist_border, dist_popcenter, dist_admhq

Bioclimatic: af_bio_1_x, af_bio_8_x, af_bio_12_x, af_bio_13_x, af_bio_16_x

Vegetation indices: evimax_avg, grn_avg, sen_avg

Geospatial: lat_dd_mod, lon_dd_mod

Other derived metrics: h2021_eviarea, h2021_evimax, etc.

Target: rainfall (mm)

Data Preprocessing

Outlier Treatment: Capped extreme values using IQR method.

Log Transformation: Reduced skewness and stabilized variance for numeric features.

Categorical Encoding: One-hot encoded ssa_aez09 and landcov.

Feature Scaling: Standardized all numeric features with StandardScaler.

Multicollinearity Handling: Removed features with high Variance Inflation Factor (VIF).

Models Used
1️⃣ Linear Regression

Captures linear relationships between features and rainfall.

Metrics:

Train R²: 0.9970

Test R²: 0.9921

MAE: 23.83

RMSE: 37.34

Pros: Fast, interpretable

Cons: Cannot capture non-linear relationships

2️⃣ Decision Tree Regressor (max_depth=5)

Handles non-linear relationships between features.

Metrics:

Train R²: 0.9986

Test R²: 0.9978

MAE: 15.67

RMSE: 19.83

Pros: Visualizable, interpretable

Cons: Single tree may overfit without depth control

3️⃣ Random Forest Regressor (200 trees, default)

Ensemble of decision trees; reduces variance and overfitting.

Metrics:

Train R²: 0.9999

Test R²: 0.9991

MAE: 7.57

RMSE: 12.39

Pros: High accuracy, robust to overfitting, handles high-dimensional data

Cons: Less interpretable than single trees, computationally heavier

4️⃣ Random Forest Regressor (100 trees, max_depth=10, max_features='sqrt')

Tuned Random Forest for regularization.

Metrics:

Train R²: 0.9969

Test R²: 0.9790

MAE: 38.26

RMSE: 61.03

Observation: Slight underfitting due to reduced model complexity

Model Comparison
Model	R² (Train)	R² (Test)	MAE	RMSE	Notes
Linear Regression	0.9970	0.9921	23.83	37.34	Linear, interpretable
Decision Tree (max_depth=5)	0.9986	0.9978	15.67	19.83	Non-linear, visualizable
Random Forest (200 trees)	0.9999	0.9991	7.57	12.39	Best performance, robust
Random Forest (100 trees, tuned)	0.9969	0.9790	38.26	61.03	Slight underfitting

✅ Best Model: Random Forest Regressor with 200 trees (highest accuracy and lowest error).

Key Benefits of Selected Model

High predictive accuracy (R² > 0.99)

Robust to feature collinearity and outliers

Captures non-linear feature interactions

Suitable for high-dimensional datasets

Reduces overfitting via ensemble averaging

Recommendations

Deploy Random Forest Regressor for operational rainfall prediction.

Collect more recent environmental data to further improve accuracy.

Use feature importance plots for interpretability.

Integrate the model into a Flask-based web application for real-time predictions.

Conclusion

This project successfully develops a robust model for rainfall prediction. The Random Forest Regressor (200 trees) outperformed all other models in accuracy and robustness, making it suitable for ESS operational use.
