from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import *
from .scoring.data_loader import DataLoader
from .scoring.customer_scorer import CustomerScorer
from .scoring.churn_predictor import ChurnPredictor
from .scoring.clv_estimator import CLVEstimator
from .scoring.survival_analyzer import SurvivalAnalyzer

app = FastAPI(title="Customer Scoring API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_loader = DataLoader()
customer_scorer = CustomerScorer(data_loader)

@app.on_event("startup")
async def startup_event():
    """Load data and models on startup"""
    data_loader.load_data()
    data_loader.load_models()
    customer_scorer.initialize()

@app.get("/")
async def root():
    return {"message": "Customer Scoring API is running"}

@app.post("/score_customer", response_model=CustomerScoreResponse)
async def score_customer(request: CustomerScoreRequest):
    """Unified customer scoring"""
    try:
        result = customer_scorer.score_customer(request.customer_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_churn", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    """Churn prediction"""
    try:
        result = customer_scorer.churn_predictor.predict_churn(
            request.customer_id, 
            request.horizon_days
        )
        result["customer_id"] = request.customer_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_survival", response_model=SurvivalPredictionResponse)
async def predict_survival(request: SurvivalPredictionRequest):
    """Survival curve prediction"""
    try:
        result = customer_scorer.survival_analyzer.predict_survival(request.customer_id)
        result["customer_id"] = request.customer_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/estimate_clv", response_model=CLVEstimationResponse)
async def estimate_clv(request: CLVEstimationRequest):
    """CLV estimation"""
    try:
        if request.method == "bgnbd":
            clv = customer_scorer.clv_estimator.estimate_clv_bgnbd(request.customer_id)
        else:
            clv = customer_scorer.clv_estimator.estimate_clv_survival(request.customer_id)
        
        return {
            "customer_id": request.customer_id,
            "method": request.method,
            "clv": clv,
            "horizon_months": 12
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank_customers_for_retention", response_model=RetentionRankingResponse)
async def rank_customers_for_retention(request: RetentionRankingRequest):
    """Rank customers for retention campaigns"""
    try:
        # Score all customers
        all_customers = data_loader.customers["customer_id"].tolist()
        scores = []
        
        for cust_id in all_customers[:request.top_k * 2]:  # Sample more than needed
            try:
                score = customer_scorer.score_customer(cust_id)
                scores.append({
                    "customer_id": cust_id,
                    "churn_probability": score["churn_probability"],
                    "clv": score["clv_bgnbd"],
                    "priority_score": score["churn_probability"] * score["clv_bgnbd"]
                })
            except:
                continue
        
        # Sort by strategy
        if request.strategy == "high_clv_high_churn":
            scores.sort(key=lambda x: x["priority_score"], reverse=True)
        elif request.strategy == "high_churn":
            scores.sort(key=lambda x: x["churn_probability"], reverse=True)
        else:  # low_p_alive
            scores.sort(key=lambda x: (1 - x["churn_probability"]))
        
        return {
            "strategy": request.strategy,
            "customers": scores[:request.top_k]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))