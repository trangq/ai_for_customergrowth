import pandas as pd
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data

class CLVEstimator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.bgnbd_model = None
        self.gg_model = None
        
    def initialize(self):
        """Initialize CLV models"""
        self.bgnbd_model = self.data_loader.models.get("bgnbd")
        self.gg_model = self.data_loader.models.get("gamma_gamma")
        
    def estimate_clv_bgnbd(self, customer_id: str, horizon_months: int = 12) -> float:
        """Estimate CLV using BG-NBD + Gamma-Gamma"""
        customer_data = self.data_loader.get_customer_data(customer_id)
        txns = customer_data["transactions"]
        
        if txns.empty or len(txns) < 2:
            return 0.0
        
        # Prepare summary data
        end_date = pd.Timestamp.now()
        summary = summary_data_from_transaction_data(
            txns,
            customer_id_col="customer_id",
            datetime_col="transaction_date",
            monetary_value_col="amount",
            observation_period_end=end_date
        )
        
        # Predict transactions
        horizon_days = horizon_months * 30
        expected_txns = self.bgnbd_model.conditional_expected_number_of_purchases_up_to_time(
            horizon_days,
            summary["frequency"].values[0],
            summary["recency"].values[0],
            summary["T"].values[0]
        )
        
        # Predict monetary value
        expected_monetary = self.gg_model.conditional_expected_average_profit(
            summary["frequency"].values[0],
            summary["monetary_value"].values[0]
        )
        
        clv = expected_txns * expected_monetary
        return float(clv)
    
    def estimate_clv_survival(self, customer_id: str, horizon_months: int = 12) -> float:
        """Estimate CLV using survival analysis"""
        # This would use the CoxPH model
        # Simplified version here
        clv_bgnbd = self.estimate_clv_bgnbd(customer_id, horizon_months)
        
        # Apply survival discount (placeholder logic)
        survival_prob = 0.7  # Would come from actual survival model
        
        return float(clv_bgnbd * survival_prob)