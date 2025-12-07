# 🌬️ Wind Power Prediction: Time-Series Regression Pipeline for Wind Turbine Data

This project implements a **modular machine-learning pipeline** for predicting wind-turbine power output using wind speed and wind direction.  
It focuses on robust preprocessing of real-world time-series sensor data, cyclic feature engineering, model training, hyperparameter tuning, and MLflow-based experiment tracking.  
The full solution is implemented in **Python (Pandas, NumPy, Scikit-Learn, MLflow)** and organized inside a modular Jupyter Notebook.

---

## 🚀 Project Overview

The pipeline is designed to handle *inconsistent, irregularly sampled time-series data*.  
Key capabilities include:

- Cleaning, resampling, and interpolating wind and power measurements  
- Cyclic wind-direction encoding (radians, sine, cosine)  
- Timestamp alignment and validation across heterogeneous datasets  
- Leakage-aware splitting using **TimeSeriesSplit**  
- Modular model pipelines using Scikit-Learn  
- Hyperparameter tuning via **GridSearchCV**  
- Full experiment tracking with MLflow  

The goal is to understand how wind features influence turbine power output and to identify the most reliable predictive model.

---

## 🧪 Key Features

- Time-series preprocessing pipeline (resampling, interpolation, normalization)  
- Automatic handling of missing data and inconsistent intervals  
- Cyclic transformation of wind-direction (radians, sin, cos)  
- Support for Linear Regression, Random Forest, and Gradient Boosting  
- Time-aware cross-validation (no data leakage)  
- Hyperparameter tuning  
- MLflow experiment logging (metrics, parameters, artifacts)  
- Comparison across **MAE, MSE, RMSE, R²**

---

## 🔍 Exploratory Data Analysis (EDA)

The dataset consists of:

- **Wind data** sampled roughly every 3 hours  
- **Power output data** sampled every minute  

To unify them:

### EDA Steps
- Checked sampling consistency using index diff analysis  
- Normalized both datasets to **3-hour intervals** using `resample()`  
- Applied `interpolate()` to preserve natural fluctuations  
- Avoided constant/mean filling to prevent flattening trends  
- Merged datasets via outer join on aligned timestamps  
- Verified all timestamps were stored in UTC  

This produced a clean, consistent time-series suitable for modeling.

---

## 🌪 Wind Direction Feature Engineering

Wind direction is inherently cyclic (0° = 360°).  
To capture this relationship, the pipeline derives:

1. **Radians**  
2. **Sin(direction)**  
3. **Cos(direction)**  

These features outperform one-hot encoding, which introduces unnecessary dimensions and noise for directional data.

---

## 🧠 Machine Learning: Splitting & Pipeline Design

### Why Not a Standard Train/Test Split?
A random split caused temporal leakage (training on future data).  
This broke trend dependencies and produced inflated scores.

### Correct Approach: **TimeSeriesSplit**
- 3 folds for 90-day and 180-day datasets  
- 4 folds for 365-day datasets  
- Optional temporal gap (24 hours) to prevent immediate leakage  
- Maintains chronological order  
- Enables season-aware validation for long datasets  

---

## 🏗 Model Pipelines

Models are wrapped in Scikit-Learn pipelines:

```python
pipelines = {
    "linr": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    "gb": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor())
    ]),
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor())
    ]),
}

# Hyperparameter Grids
param_grid_dictionary = {
    "linr": {
        "regressor__fit_intercept": [True, False],
    },
    "gb": {
        "regressor__n_estimators": [50, 100],
        "regressor__learning_rate": [0.01, 0.1],
        "regressor__max_depth": [3, 5],
    },
    "rf": {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [5, 10],
        "regressor__min_samples_split": [2, 5],
    },
}
```

## Key Findings

Experiments revealed that:

* Gradient Boosting consistently performed best across all metrics

* Cyclic wind-direction features helped short-term predictions, with diminishing effect at full-year scale

* Resampling to 3-hour intervals plus interpolation reduced noise while retaining natural variability

* TimeSeriesSplit significantly improved evaluation stability

* Ensemble models outperformed linear models under noisy seasonal conditions

## What I Learned

This project improved my understanding of:

* Time-series engineering and leakage-aware splitting

* Cyclic feature engineering for directional data

* Ensemble regression and hyperparameter optimization

* Modular ML pipeline design

* Experiment tracking and reproducibility with MLflow

