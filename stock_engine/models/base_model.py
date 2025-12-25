"""Base classes for models."""

from abc import ABC, abstractmethod
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Initialize base model.
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost', 'catboost')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        pass
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']

