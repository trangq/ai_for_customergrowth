import pandas as pd
import joblib
from pathlib import Path
from typing import Dict

class DataLoader:
    def __init__(self, base_path: str = r"d:\boothcamp_customergrowth"):
        self.base_path = Path(base_path)
        self.customers = None
        self.transactions = None
        self.models = {}
        
    def load_data(self):
        """Load customer and transaction data"""
        self.customers = pd.read_csv(self.base_path / "customers.csv")
        self.transactions = pd.read_csv(self.base_path / "transactions.csv")
        self.transactions["transaction_date"] = pd.to_datetime(self.transactions["transaction_date"])
        self.customers["signup_date"] = pd.to_datetime(self.customers["signup_date"])
        
    def load_models(self):
        """Load all trained models"""
        model_files = {
            "logistic_regression": "models/logistic_regression_model.pkl",
            "bgnbd": "models/bgnbd_model.pkl",
            "gamma_gamma": "models/gamma_gamma_model.pkl",
            "coxph": "models/coxph_model.pkl"
        }
        
        for name, filename in model_files.items():
            try:
                self.models[name] = joblib.load(self.base_path / filename)
            except FileNotFoundError:
                print(f"Warning: {filename} not found")
                
    def get_customer_data(self, customer_id: str) -> Dict:
        """Get data for a specific customer"""
        customer = self.customers[self.customers["customer_id"] == customer_id]
        customer_txns = self.transactions[self.transactions["customer_id"] == customer_id]
        
        return {
            "customer": customer.to_dict('records')[0] if not customer.empty else None,
            "transactions": customer_txns
        }