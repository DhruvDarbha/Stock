"""Evaluation metrics for models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import spearmanr


class Evaluator:
    """Evaluate model performance."""
    
    @staticmethod
    def evaluate_market_model(
        predictions: pd.Series,
        actuals: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate market model predictions.
        
        Args:
            predictions: Predicted excess returns
            actuals: Actual excess returns
        
        Returns:
            Dictionary of metrics
        """
        # Align indices
        common_idx = predictions.index.intersection(actuals.index)
        pred = predictions[common_idx]
        act = actuals[common_idx]
        
        if len(pred) == 0:
            return {}
        
        # Correlation
        correlation = pred.corr(act)
        
        # Rank correlation (Spearman)
        rank_corr, _ = spearmanr(pred, act)
        
        # MSE, MAE
        mse = ((pred - act) ** 2).mean()
        mae = (pred - act).abs().mean()
        
        # Hit rate (top decile)
        top_decile_threshold = pred.quantile(0.9)
        top_decile_pred = pred[pred >= top_decile_threshold]
        if len(top_decile_pred) > 0:
            top_decile_actuals = act[top_decile_pred.index]
            hit_rate = (top_decile_actuals > act.median()).mean()
        else:
            hit_rate = 0.0
        
        # Calibration (for downside risk - simplified)
        # TODO: Implement proper calibration metrics
        
        return {
            'correlation': correlation,
            'rank_correlation': rank_corr,
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'hit_rate_top_decile': hit_rate,
            'n_samples': len(pred)
        }
    
    @staticmethod
    def evaluate_behavior_model(
        predictions: pd.Series,
        actuals: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate behavior model (binary classification).
        
        Args:
            predictions: Predicted probabilities
            actuals: Actual binary outcomes (0 or 1)
        
        Returns:
            Dictionary of metrics
        """
        # Align indices
        common_idx = predictions.index.intersection(actuals.index)
        pred = predictions[common_idx]
        act = actuals[common_idx]
        
        if len(pred) == 0:
            return {}
        
        # Binary predictions (threshold = 0.5)
        pred_binary = (pred >= 0.5).astype(int)
        
        # Accuracy
        accuracy = (pred_binary == act).mean()
        
        # Precision, recall, F1
        tp = ((pred_binary == 1) & (act == 1)).sum()
        fp = ((pred_binary == 1) & (act == 0)).sum()
        fn = ((pred_binary == 0) & (act == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC (simplified - would use sklearn in production)
        # TODO: Use sklearn.metrics.roc_auc_score
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_samples': len(pred)
        }

