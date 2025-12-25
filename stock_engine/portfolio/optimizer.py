"""Portfolio optimization module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import date

from ..config import get_config
from ..models import MarketModel, BehaviorModel
from ..events import EventEngine


class PortfolioOptimizer:
    """Optimize portfolio weights based on predictions and constraints."""
    
    def __init__(
        self,
        market_model: MarketModel,
        behavior_model: Optional[BehaviorModel] = None,
        event_engine: Optional[EventEngine] = None
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            market_model: Trained market prediction model
            behavior_model: Optional behavior model for follow probability
            event_engine: Optional event engine for risk adjustments
        """
        self.market_model = market_model
        self.behavior_model = behavior_model
        self.event_engine = event_engine
        self.config = get_config()
    
    def compute_target_weights(
        self,
        tickers: List[str],
        current_positions: Dict[str, float],
        user_id: Optional[int] = None,
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Compute target portfolio weights for given tickers.
        
        Args:
            tickers: List of tickers to consider
            current_positions: Dict of ticker -> current weight
            user_id: Optional user ID for behavior modeling
            as_of_date: Date for predictions
        
        Returns:
            DataFrame with columns: ticker, current_weight, target_weight, score
        """
        as_of_date = as_of_date or date.today()
        
        # Get predictions from market model
        from ..features import StockFeatureEngine
        feature_engine = StockFeatureEngine()
        
        features_df = feature_engine.compute_features_batch(tickers, as_of_date)
        
        if features_df.empty:
            return pd.DataFrame()
        
        # Predict excess returns
        excess_returns = self.market_model.predict(features_df)
        risk_scores = self.market_model.predict_risk(features_df)
        
        # Adjust for events if event engine available
        risk_multipliers = {}
        if self.event_engine:
            for ticker in tickers:
                should_apply, multiplier = self.event_engine.should_apply_risk_multiplier(
                    ticker, as_of_date
                )
                if should_apply:
                    risk_multipliers[ticker] = multiplier
                    risk_scores[ticker] *= multiplier
        
        # Compute risk-adjusted scores
        # Sharpe-like metric: expected_return / risk
        scores = excess_returns / (risk_scores + 0.01)  # Add small epsilon to avoid division by zero
        
        # Adjust for user behavior if behavior model available
        if self.behavior_model and user_id:
            adjusted_scores = {}
            for ticker in tickers:
                if ticker not in scores.index:
                    continue
                
                base_score = scores[ticker]
                
                # Estimate follow probability (simplified - would need recommendation features)
                rec_features = {
                    'recommended_action_buy': 1.0,
                    'recommended_action_sell': 0.0,
                    'recommended_shares': 0.0,
                    'recommended_weight': 0.05
                }
                
                try:
                    follow_prob = self.behavior_model.predict_follow_probability(
                        user_id, rec_features
                    )
                    # Weight score by follow probability
                    adjusted_scores[ticker] = base_score * follow_prob
                except:
                    adjusted_scores[ticker] = base_score
            
            scores = pd.Series(adjusted_scores)
        
        # Normalize scores to weights under constraints
        target_weights = self._optimize_weights(
            scores,
            current_positions,
            tickers
        )
        
        # Ensure all tickers have predictions (default to 0.0 if missing)
        # This happens if feature computation failed for some tickers
        for ticker in tickers:
            if ticker not in excess_returns:
                excess_returns[ticker] = 0.0
            if ticker not in risk_scores:
                risk_scores[ticker] = 0.15
            if ticker not in scores:
                scores[ticker] = 0.0
        
        # Create result DataFrame
        result = pd.DataFrame({
            'ticker': tickers,
            'current_weight': [current_positions.get(t, 0.0) for t in tickers],
            'target_weight': [target_weights.get(t, 0.0) for t in tickers],
            'expected_excess_return': [excess_returns.get(t, 0.0) for t in tickers],
            'risk_score': [risk_scores.get(t, 0.15) for t in tickers],
            'score': [scores.get(t, 0.0) for t in tickers]
        })
        
        return result
    
    def _optimize_weights(
        self,
        scores: pd.Series,
        current_positions: Dict[str, float],
        tickers: List[str]
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights under constraints.
        
        Uses a simplified approach:
        1. Rank by score
        2. Allocate weights subject to constraints
        3. Respect max single stock weight, sector caps, etc.
        
        TODO: Replace with proper optimization (e.g., mean-variance, risk parity)
        """
        # Get constraints
        max_single_stock = self.config.get('portfolio.max_single_stock_weight', 0.10)
        min_weight = self.config.get('portfolio.min_trade_size', 0.005)
        
        # Sort by score descending
        sorted_scores = scores.sort_values(ascending=False)
        
        target_weights = {}
        
        # Filter positive scores
        positive_scores = sorted_scores[sorted_scores > 0]
        
        if len(positive_scores) == 0:
            # All scores negative/zero - allocate equally among top tickers
            print("  ⚠️  All scores are non-positive, using equal allocation")
            num_holdings = min(5, len(tickers))  # Hold top 5 stocks equally
            weight_per_stock = (1.0 - 0.1) / num_holdings  # Reserve 10% cash
            for i, ticker in enumerate(tickers[:num_holdings]):
                target_weights[ticker] = weight_per_stock
            for ticker in tickers:
                if ticker not in target_weights:
                    target_weights[ticker] = 0.0
            return target_weights
        
        total_allocated = 0.0
        
        # Allocate to top stocks with positive scores
        # Use softmax-like allocation: proportional to scores
        score_sum = positive_scores.sum()
        
        for ticker, score in positive_scores.items():
            if ticker not in tickers:
                continue
            
            remaining_capacity = 1.0 - total_allocated
            if remaining_capacity <= min_weight:
                break
            
            # Allocate weight proportional to score, capped at max_single_stock
            proportional_weight = (score / score_sum) * 0.9  # Use 90% of portfolio
            target_weight = min(proportional_weight, max_single_stock, remaining_capacity)
            
            # Only allocate if above minimum
            if target_weight >= min_weight:
                target_weights[ticker] = target_weight
                total_allocated += target_weight
        
        # Normalize to use 90% of portfolio (keep 10% cash)
        if total_allocated > 0:
            scale = 0.9 / total_allocated  # Scale to 90% allocation
            for ticker in target_weights:
                target_weights[ticker] *= scale
        
        # Fill in zeros for other tickers
        for ticker in tickers:
            if ticker not in target_weights:
                target_weights[ticker] = 0.0
        
        return target_weights

