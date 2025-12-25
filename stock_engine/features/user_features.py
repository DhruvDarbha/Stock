"""User-level feature engineering."""

import pandas as pd
import numpy as np
from typing import Dict, Any

from ..data import get_user_data_store


class UserFeatureEngine:
    """Engine for computing user-level features."""
    
    def __init__(self):
        """Initialize user feature engine."""
        self.user_store = get_user_data_store()
    
    def compute_features(self, user_id: int) -> Dict[str, float]:
        """
        Compute all features for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            Dictionary of feature name -> value
        """
        user = self.user_store.get_user(user_id)
        if not user:
            return self._empty_features()
        
        features = {}
        
        # Static features
        features.update(self._compute_static_features(user))
        
        # Dynamic/behavioral features
        features.update(self._compute_behavioral_features(user_id))
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            'risk_profile_conservative': 0.0,
            'risk_profile_moderate': 1.0,
            'risk_profile_aggressive': 0.0,
            'horizon_short': 0.0,
            'horizon_medium': 1.0,
            'horizon_long': 0.0,
            'follow_rate': 0.0,
            'avg_holding_period_days': 0.0,
            'max_drawdown_tolerated': 0.0,
            'volatility_sensitivity': 0.0,
            'preferred_turnover': 0.0
        }
    
    def _compute_static_features(self, user: Dict[str, Any]) -> Dict[str, float]:
        """Compute static user features."""
        features = {}
        
        # Risk profile (one-hot encoded)
        risk_profile = user.get('risk_profile', 'moderate').lower()
        features['risk_profile_conservative'] = 1.0 if risk_profile == 'conservative' else 0.0
        features['risk_profile_moderate'] = 1.0 if risk_profile == 'moderate' else 0.0
        features['risk_profile_aggressive'] = 1.0 if risk_profile == 'aggressive' else 0.0
        
        # Investment horizon (one-hot encoded)
        horizon = user.get('investment_horizon', 'medium').lower()
        features['horizon_short'] = 1.0 if horizon == 'short' else 0.0
        features['horizon_medium'] = 1.0 if horizon == 'medium' else 0.0
        features['horizon_long'] = 1.0 if horizon == 'long' else 0.0
        
        # Constraints
        constraints = user.get('constraints', {})
        features['max_single_stock_weight'] = constraints.get('max_single_stock_weight', 0.10)
        features['max_sector_weight'] = constraints.get('max_sector_weight', 0.30)
        
        return features
    
    def _compute_behavioral_features(self, user_id: int) -> Dict[str, float]:
        """Compute behavioral features from user history."""
        features = {}
        
        # Get behavior stats
        stats = self.user_store.get_user_behavior_stats(user_id)
        
        features['follow_rate'] = stats.get('follow_rate', 0.0)
        features['avg_holding_period_days'] = stats.get('avg_holding_period_days', 0.0)
        features['max_drawdown_tolerated'] = stats.get('max_drawdown_tolerated', 0.0)
        features['volatility_sensitivity'] = stats.get('volatility_sensitivity', 0.0)
        
        # Compute preferred turnover from action history
        actions = self.user_store.get_user_actions(user_id)
        if not actions.empty and len(actions) > 1:
            # Estimate turnover from action frequency
            actions_per_year = len(actions) / (actions['timestamp'].max() - actions['timestamp'].min()).days * 365
            features['preferred_turnover'] = min(actions_per_year / 12, 1.0)  # Normalize to 0-1
        else:
            features['preferred_turnover'] = 0.5  # Default moderate turnover
        
        return features
    
    def compute_features_batch(self, user_ids: list) -> pd.DataFrame:
        """
        Compute features for multiple users.
        
        Returns:
            DataFrame with user_id as index and features as columns
        """
        results = []
        
        for user_id in user_ids:
            try:
                features = self.compute_features(user_id)
                features['user_id'] = user_id
                results.append(features)
            except Exception as e:
                print(f"Error computing features for user {user_id}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df.set_index('user_id', inplace=True)
        return df

