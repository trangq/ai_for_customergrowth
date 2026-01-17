import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ChurnPredictor:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = None
        
    def initialize(self):
        """Initialize the churn prediction model"""
        self.model = self.data_loader.models.get("logistic_regression")
        
    def predict_churn(self, customer_id: str, horizon_days: int = 60) -> dict:
        """Predict churn probability for a customer"""
        if self.model is None:
            raise ValueError("Model not initialized")
            
        # Get customer features
        features = self._extract_features(customer_id)
        
        # Predict
        churn_prob = self.model.predict_proba(features)[:, 1][0]
        
        # Categorize risk
        if churn_prob >= 0.7:
            label = "high_risk"
        elif churn_prob >= 0.4:
            label = "medium_risk"
        else:
            label = "low_risk"
            
        return {
            "churn_probability": float(churn_prob),
            "churn_label": label
        }
    
    def _extract_features(self, customer_id: str) -> pd.DataFrame:
        """Extract RFM and trend features for prediction"""
        snapshot_date = pd.Timestamp.now()
        
        customer_data = self.data_loader.get_customer_data(customer_id)
        txns = customer_data["transactions"]
        
        if txns.empty:
            # Return default features for customers with no transactions
            return pd.DataFrame({
                "Recency": [999],
                "Frequency": [0],
                "Monetary": [0],
                "freq_recent": [0],
                "freq_past": [0],
                "freq_trend": [0]
            })
        
        # Calculate RFM
        last_txn = txns["transaction_date"].max()
        recency = (snapshot_date - last_txn).days
        frequency = len(txns)
        monetary = txns["amount"].sum()
        
        # Calculate frequency trend
        recent_start = snapshot_date - timedelta(days=60)
        recent_txns = txns[txns["transaction_date"] > recent_start]
        past_txns = txns[txns["transaction_date"] <= recent_start]
        
        freq_recent = len(recent_txns)
        freq_past = len(past_txns)
        freq_trend = freq_recent - freq_past
        
        return pd.DataFrame({
            "Recency": [recency],
            "Frequency": [frequency],
            "Monetary": [monetary],
            "freq_recent": [freq_recent],
            "freq_past": [freq_past],
            "freq_trend": [freq_trend]
        })