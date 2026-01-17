# Customer Scoring API - Setup & Run Guide

## Project Overview
This project implements a comprehensive customer scoring system with:
- ✅ Churn Prediction (Logistic Regression)
- ✅ RFM Analysis (Recency, Frequency, Monetary)
- ✅ BG-NBD Lifetime Value Model
- ✅ Survival Analysis (Cox PH & Weibull)
- ✅ CLV Estimation
- ✅ REST API with FastAPI

---

## Prerequisites
- Python 3.8+
- pip (Python package manager)

---

## Step 1: Install Dependencies

Open PowerShell in the project directory (`d:\boothcamp_customergrowth`) and run:

```powershell
pip install -r requirements.txt
```

This installs:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `lifetimes` - BG-NBD models
- `joblib` - Model serialization
- `pydantic` - Data validation

**Time:** ~2-3 minutes depending on internet speed

---

## Step 2: Prepare Models

Models are already trained and saved in `models/` folder:
```
models/
├── logistic_regression_model.pkl    (Churn classifier)
├── bgnbd_model.pkl                  (Lifetime value - BG-NBD)
├── gamma_gamma_model.pkl            (Monetary value estimation)
├── coxph_model.pkl                  (Survival analysis - Cox)
└── weibull_aft_model.pkl            (Survival analysis - Weibull)
```

**If models are missing:** Run `project.ipynb` to retrain them.

---

## Step 3: Run the API Server

In PowerShell (from project directory):

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

The API is now running at: **http://localhost:8000**

---

## Step 4: Test the API

### Option A: Interactive Docs (Recommended)
Open in browser: **http://localhost:8000/docs**

You'll see Swagger UI with all endpoints and try-it-out buttons!

### Option B: PowerShell Command

Test churn prediction:
```powershell
$body = @{"customer_id" = "C00000"} | ConvertTo-Json
$uri = "http://localhost:8000/score_customer"
Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType "application/json"
```

### Option C: Python Client (example_client.py)

```powershell
python example_client.py
```

---

## API Endpoints

### 1. **Unified Customer Score** 
```
POST /score_customer
```
**Input:**
```json
{"customer_id": "C00000"}
```

**Output:**
```json
{
  "customer_id": "C00000",
  "churn_probability": 0.42,
  "p_alive": 0.63,
  "expected_remaining_lifetime": 58.4,
  "clv_bgnbd": 1250,
  "clv_survival": 980
}
```

---

### 2. **Churn Prediction**
```
POST /predict_churn
```
**Input:**
```json
{
  "customer_id": "C00000",
  "horizon_days": 60
}
```

**Output:**
```json
{
  "customer_id": "C00000",
  "churn_probability": 0.42,
  "churn_label": "high_risk"
}
```

**Risk Labels:**
- `low_risk`: probability < 0.33
- `medium_risk`: 0.33 ≤ probability < 0.67
- `high_risk`: probability ≥ 0.67

---

### 3. **Survival Curve Prediction**
```
POST /predict_survival
```
**Input:**
```json
{"customer_id": "C00000"}
```

**Output:**
```json
{
  "customer_id": "C00000",
  "survival_curve": [
    {"day": 30, "prob": 0.82},
    {"day": 60, "prob": 0.65},
    {"day": 90, "prob": 0.47}
  ],
  "expected_remaining_lifetime": 58.4
}
```

---

### 4. **CLV Estimation**
```
POST /estimate_clv
```
**Input (BG-NBD method):**
```json
{
  "customer_id": "C00000",
  "method": "bgnbd"
}
```

**Input (Survival method):**
```json
{
  "customer_id": "C00000",
  "method": "survival"
}
```

**Output:**
```json
{
  "customer_id": "C00000",
  "method": "bgnbd",
  "clv": 1250,
  "horizon_months": 12
}
```

---

### 5. **Rank Customers for Retention**
```
POST /rank_customers_for_retention
```
**Input:**
```json
{
  "top_k": 100,
  "strategy": "high_clv_high_churn"
}
```

**Strategies:**
- `high_churn`: Most at-risk customers
- `high_clv`: Most valuable customers
- `high_clv_high_churn`: High value AND at-risk (Priority!)

**Output:**
```json
{
  "strategy": "high_clv_high_churn",
  "customers": [
    {
      "customer_id": "C00000",
      "churn_probability": 0.42,
      "clv": 1250,
      "priority_score": 0.78
    }
  ]
}
```

---

## Project Structure

```
d:\boothcamp_customergrowth/
├── requirements.txt              # Dependencies
├── project.ipynb                 # Training notebook
├── customers.csv                 # Customer data
├── transactions.csv              # Transaction history
├── RUN_GUIDE.md                 # This file
│
├── app/
│   ├── main.py                  # FastAPI application
│   ├── models.py                # Pydantic models (schemas)
│   └── scoring/
│       ├── data_loader.py       # Load CSV & models
│       ├── customer_scorer.py   # Unified scoring logic
│       ├── churn_predictor.py   # Classification
│       ├── clv_estimator.py     # BG-NBD & Survival CLV
│       └── survival_analyzer.py # Survival curves
│
└── models/
    ├── logistic_regression_model.pkl
    ├── bgnbd_model.pkl
    ├── gamma_gamma_model.pkl
    ├── coxph_model.pkl
    └── weibull_aft_model.pkl
```

---

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Run API: `python -m uvicorn app.main:app --reload --port 8000`
3. ✅ Open docs: `http://localhost:8000/docs`
4. ✅ Test endpoints with Swagger UI
5. ✅ Integrate with your frontend/application

---

## Questions?

- Check `MODELS_INFO.md` for detailed model information
- Review `project.ipynb` for training methodology
- Inspect individual modules in `app/scoring/` for implementation details

Happy scoring! 🚀
