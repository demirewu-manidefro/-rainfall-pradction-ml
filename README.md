Predicting Rainfall Using Machine Learning

Internship Project – Ethiopia Statistical Service

Table of Contents

Introduction

Dataset Description

Data Preprocessing

Feature Scaling

Train-Test Split

Model Development

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Feature Importance Analysis

Model Validation

Cross-Validation

Prediction on New Data

Feature Selection

Model Evaluation Metrics

Conclusion

Future Work

References

Introduction

The goal of this project is to develop a robust machine learning model for predicting rainfall based on environmental, climatic, and geographical features. Rainfall prediction is critical for:

Agriculture planning

Water resource management

Disaster preparedness

This project was conducted as part of a Data Science internship at Ethiopia Statistical Service.

The dataset contains 514 observations with 79 features, including climate, geographical distances, elevation, population density, land cover, and agro-climatic indicators.

Project Workflow:

Data preprocessing and feature scaling

Train-test splitting

Model development (Linear Regression, Decision Tree, Random Forest)

Feature importance analysis

Feature selection

Model evaluation using multiple metrics

Prediction on new samples

Dataset Description

The dataset contains 514 observations and 79 features, including:

Geospatial distances: Distance to road, market, border, population centers, administrative headquarters

Climatic indicators: af_bio variables (temperature & precipitation metrics), wetQ_avg, afmnslp_pct

Elevation: srtm_1k, srtm_1k_log

Land cover variables: Cropland, forests, shrubs, herbaceous vegetation

Population indicators: dist_popcenter, pct_urban_cluster

Target variable: rainfall (in mm)

Preprocessing: Some features were log-transformed (rainfall_log, af_bio_*_log, dist_*_log) to normalize skewed distributions.

Data Preprocessing
Feature Scaling

All features were standardized using StandardScaler to have mean = 0 and standard deviation = 1.

from sklearn.preprocessing import StandardScaler

X = df_encoded.drop('rainfall', axis=1)
y = df_encoded['rainfall']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)


Result: Scaled dataset with shape (514, 80)

Train-Test Split

Data was split into training (80%) and testing (20%) sets:

from sklearn.model_selection import train_test_split

X = df_scaled.drop('rainfall', axis=1)
y = df_scaled['rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Training set: (411, 79)

Testing set: (103, 79)

Model Development
Linear Regression

Objective: Serve as a baseline model

Results:

R² (Train) ≈ 0.9970

R² (Test) ≈ 0.9921

MAE ≈ 23.83 mm

RMSE ≈ 37.33 mm

Observation: Slight overfitting; handles linear relationships reasonably well

Decision Tree Regressor

Configuration: max_depth=5

Results:

R² (Train) ≈ 0.9986

R² (Test) ≈ 0.9978

MAE ≈ 15.67 mm

RMSE ≈ 19.83 mm

Observation: Captures non-linear patterns better than Linear Regression

Random Forest Regressor

Method: Ensemble of multiple decision trees

Experiments:

Full model (200 trees)

Train R² ≈ 0.9999

Test R² ≈ 0.9991

MAE ≈ 7.57 mm, RMSE ≈ 12.39 mm

Tuned model (max_depth=10, max_features='sqrt')

Test R² ≈ 0.9789, MAE ≈ 38.25 mm

Top 20 features (selected via permutation importance)

Test R² ≈ 0.9996, MAE ≈ 5.07 mm, RMSE ≈ 7.94 mm

Observation: Random Forest provides the best performance, capturing non-linearities effectively

Feature Importance Analysis
Standard RF Feature Importance
Feature	Importance
rainfall_log	0.996
af_bio_16_x	0.00063
af_bio_16_x_log	0.00048
af_bio_13_x	0.00028
af_bio_13_x_log	0.00026

Observation: rainfall_log dominates predictive power

Permutation Importance

Identifies true contribution of each feature

Top 20 features selected for reduced model improve interpretability without sacrificing accuracy

Model Validation
Cross-Validation

5-fold CV on top 20 features:

Mean R² ≈ 0.99927

CV MAE ≈ 5.92 mm

Observation: Model is stable and generalizes well

Prediction on New Data
sample_data = {col: [0.5] for col in top_20_features}
sample_df = pd.DataFrame(sample_data)

predicted_rainfall = rf.predict(sample_df)[0]


Predicted rainfall: 1138.38 mm

Actual rainfall: 1027 mm

Error: 111.38 mm

Another example:

Predicted: 1882.55 mm

Actual: 1850 mm

Error ≈ 32.55 mm → very accurate

Feature Selection

Reduced from 79 → 20 features using permutation importance

Benefits:

Faster training & prediction

Improved interpretability

Slightly improved test accuracy (R² ≈ 0.9996)

Top features include:
rainfall_log, af_bio_16_x, af_bio_16_x_log, af_bio_13_x_log, af_bio_13_x, cropshare, dist_market_log, dist_road_log, …

Model Evaluation Metrics
Metric	Linear Regression	Decision Tree	Random Forest (Top 20)
R² (Train)	0.9903	0.9986	0.9996
R² (Test)	0.9895	0.9978	0.9996
MAE	31.71 mm	15.67 mm	5.07 mm
RMSE	43.14 mm	19.83 mm	7.94 mm

Observation: Random Forest (Top 20 features) is the most accurate and reliable

Conclusion

Best model: Random Forest Regressor

Advantages: High accuracy (R² > 0.999), low error (MAE ≈ 5 mm), strong generalization

Recommendations:

Deploy Random Forest for operational rainfall prediction

Monitor feature drift and update model periodically

Explore additional environmental variables (e.g., soil moisture, satellite indices)

Use top 20 features for simplified reporting

Future Work

Hyperparameter tuning (GridSearchCV / Bayesian optimization)

Deploy as a web-based prediction tool

Integrate with geospatial data for regional rainfall mapping

Explore other ensemble models (XGBoost, LightGBM) for comparison

References

Scikit-learn Documentation

Breiman, L. (2001). Random Forests. Machine Learning.

Ethiopia Statistical Service (2023). Climate and Agro-E
