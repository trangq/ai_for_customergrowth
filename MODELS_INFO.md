# Saved Models

All trained models are saved in the `models/` folder:

## 1. **logistic_regression_model.pkl**
- **Type**: Logistic Regression Classifier
- **Purpose**: Churn prediction (60-day inactivity window)
- **Input Features**: Recency, Frequency, Monetary, freq_recent, freq_past, freq_trend
- **Output**: Churn probability (0-1) and binary classification

## 2. **bgnbd_model.pkl**
- **Type**: Beta-Geometric/Negative Binomial Distribution (BetaGeoFitter)
- **Purpose**: Customer lifecycle modeling
- **Predicts**:
  - P(alive): Probability customer is still active
  - Expected number of transactions in future period
  - Used for CLV estimation with Gamma-Gamma

## 3. **gamma_gamma_model.pkl**
- **Type**: Gamma-Gamma Model (GammaGammaFitter)
- **Purpose**: Expected monetary value estimation
- **Input**: Customer frequency and historical monetary value
- **Output**: Expected average transaction value
- **Note**: Only fitted on customers with frequency > 0

## 4. **coxph_survival_model.pkl**
- **Type**: Cox Proportional Hazards Model (CoxPHFitter)
- **Purpose**: Time-to-churn survival analysis
- **Input Features**: Recency, Frequency, Monetary
- **Output**:
  - Survival probability at specific time points
  - Expected remaining lifetime (median/mean)
  - Hazard ratios (feature importance)

## 5. **weibull_aft_model.pkl**
- **Type**: Weibull Accelerated Failure Time Model (WeibullAFTFitter)
- **Purpose**: Alternative survival analysis model
- **Input Features**: Recency, Frequency, Monetary
- **Output**:
  - Parametric survival curves
  - Expected remaining lifetime
  - Time-to-event predictions

## Loading Models

```python
import joblib

models_dir = r'd:\boothcamp_customergrowth\models'

# Load any model
bgf = joblib.load(f'{models_dir}/bgnbd_model.pkl')
ggf = joblib.load(f'{models_dir}/gamma_gamma_model.pkl')
cph = joblib.load(f'{models_dir}/coxph_survival_model.pkl')
weibull = joblib.load(f'{models_dir}/weibull_aft_model.pkl')
lr = joblib.load(f'{models_dir}/logistic_regression_model.pkl')
```

## Relationship Between Models

- **BG-NBD + Gamma-Gamma**: Probabilistic approach for CLV
  - BG-NBD estimates frequency and P(alive)
  - Gamma-Gamma estimates monetary value
  - Combined: CLV = E[N] × E[M] × P(alive)

- **Survival Analysis (CoxPH / Weibull)**: Time-to-event approach
  - Estimates time to churn
  - Provides expected remaining lifetime
  - More interpretable with hazard ratios

- **Logistic Regression**: Classification approach
  - Binary churn prediction
  - Probability of churn in next 60 days
  - Simple and interpretable

## Feature Requirements

All models use these 6 core features:
1. **Recency**: Days since last transaction (at observation point)
2. **Frequency**: Number of transactions (up to observation point)
3. **Monetary**: Total transaction amount (up to observation point)
4. **freq_recent**: Transactions in last 60 days
5. **freq_past**: Transactions before that (120-180 days back)
6. **freq_trend**: Recent minus past (change in activity)

**Observation Point (T)**: October 31, 2025
**Prediction Horizon**: 60 days forward
**Churn Definition**: No transaction from Nov 1 - Dec 31, 2025
