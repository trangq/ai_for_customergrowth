from .churn_predictor import ChurnPredictor
from .clv_estimator import CLVEstimator
from .survival_analyzer import SurvivalAnalyzer

class CustomerScorer:
    def __init__(self, data_loader):
        self.churn_predictor = ChurnPredictor(data_loader)
        self.clv_estimator = CLVEstimator(data_loader)
        self.survival_analyzer = SurvivalAnalyzer(data_loader)
        
    def initialize(self):
        """Initialize all scorers"""
        self.churn_predictor.initialize()
        self.clv_estimator.initialize()
        self.survival_analyzer.initialize()
        
    def score_customer(self, customer_id: str) -> dict:
        """Unified customer scoring"""
        # Get churn prediction
        churn_result = self.churn_predictor.predict_churn(customer_id)
        
        # Get survival analysis
        survival_result = self.survival_analyzer.predict_survival(customer_id)
        
        # Get CLV estimates
        clv_bgnbd = self.clv_estimator.estimate_clv_bgnbd(customer_id)
        clv_survival = self.clv_estimator.estimate_clv_survival(customer_id)
        
        # Calculate P(alive) from BG-NBD model (simplified)
        p_alive = 1 - churn_result["churn_probability"]
        
        return {
            "customer_id": customer_id,
            "churn_probability": churn_result["churn_probability"],
            "p_alive": p_alive,
            "expected_remaining_lifetime": survival_result["expected_remaining_lifetime"],
            "clv_bgnbd": clv_bgnbd,
            "clv_survival": clv_survival
        }