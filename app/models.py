from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

class CustomerScoreRequest(BaseModel):
    customer_id: str

class CustomerScoreResponse(BaseModel):
    customer_id: str
    churn_probability: float
    p_alive: float
    expected_remaining_lifetime: float
    clv_bgnbd: float
    clv_survival: float

class ChurnPredictionRequest(BaseModel):
    customer_id: str
    horizon_days: int = 60

class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_label: str

class SurvivalPoint(BaseModel):
    day: int
    prob: float

class SurvivalPredictionRequest(BaseModel):
    customer_id: str

class SurvivalPredictionResponse(BaseModel):
    customer_id: str
    survival_curve: List[SurvivalPoint]
    expected_remaining_lifetime: float

class CLVEstimationRequest(BaseModel):
    customer_id: str
    method: Literal["bgnbd", "survival"] = "bgnbd"

class CLVEstimationResponse(BaseModel):
    customer_id: str
    method: str
    clv: float
    horizon_months: int = 12

class CustomerRankItem(BaseModel):
    customer_id: str
    churn_probability: float
    clv: float
    priority_score: float

class RetentionRankingRequest(BaseModel):
    top_k: int = 100
    strategy: Literal["high_clv_high_churn", "low_p_alive", "high_churn"] = "high_clv_high_churn"

class RetentionRankingResponse(BaseModel):
    strategy: str
    customers: List[CustomerRankItem]