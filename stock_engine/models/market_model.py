"""Market prediction model for stock returns."""

import pandas as pd
import numpy as np
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from .base_model import BaseModel
from ..features import StockFeatureEngine
from ..data import get_market_data_provider
from ..config import get_config


class MarketModel(BaseModel):
    """Model for predicting stock returns and risk."""
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Initialize market model.
        
        Args:
            model_type: 'lightgbm', 'xgboost', or 'catboost'
        """
        super().__init__(model_type)
        self.feature_engine = StockFeatureEngine()
        self.market_data = get_market_data_provider()
        self.config = get_config()
        self.target_horizon_days = self.config.get('models.market_model.target_horizon_days', 252)
        self.prediction_windows = self.config.get('models.market_model.prediction_window_days', [63, 126, 252])
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        **kwargs
    ):
        """
        Train the market model.
        
        Args:
            X: Feature matrix (features for each stock-date)
            y: Target vector (future excess returns)
            validation_split: Fraction to use for validation
            **kwargs: Additional training parameters
        """
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle missing values
        X = X.fillna(X.median()).fillna(0)
        
        # Split into train/validation
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        train_idx = indices[:-n_val]
        val_idx = indices[-n_val:]
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train based on model type
        if self.model_type == 'lightgbm':
            self._train_lightgbm(X_train, y_train, X_val, y_val, **kwargs)
        elif self.model_type == 'xgboost':
            self._train_xgboost(X_train, y_train, X_val, y_val, **kwargs)
        elif self.model_type == 'catboost':
            self._train_catboost(X_train, y_train, X_val, y_val, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.is_trained = True
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs
    ):
        """Train LightGBM model."""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        params.update(kwargs)
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs
    ):
        """Train XGBoost model."""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0
        }
        params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    
    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs
    ):
        """Train CatBoost model."""
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False,
            'early_stopping_rounds': 50
        }
        params.update(kwargs)
        
        self.model = CatBoostRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
    
    def predict(
        self,
        X: pd.DataFrame,
        return_std: bool = False
    ) -> pd.Series:
        """
        Predict excess returns.
        
        Args:
            X: Feature matrix
            return_std: If True, also return uncertainty estimate
        
        Returns:
            Predicted excess returns (and optionally std dev)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Handle missing values
        X = X.fillna(X.median()).fillna(0)
        
        # Ensure feature order matches training
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            X = X[self.feature_names]
        
        # Make predictions
        if self.model_type == 'lightgbm':
            predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        else:
            predictions = self.model.predict(X)
        
        pred_series = pd.Series(predictions, index=X.index)
        
        if return_std:
            # Estimate uncertainty (simplified - could use quantile regression)
            # For now, return constant uncertainty
            std_series = pd.Series([0.15] * len(predictions), index=X.index)
            return pred_series, std_series
        
        return pred_series
    
    def predict_risk(
        self,
        X: pd.DataFrame
    ) -> pd.Series:
        """
        Predict downside risk metric (e.g., expected max drawdown).
        
        TODO: Train separate risk model or use quantile regression.
        For now, use a simplified heuristic based on volatility features.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Simplified risk estimation based on volatility features
        if 'volatility_20d' in X.columns:
            risk = X['volatility_20d'] * 0.5  # Rough estimate
        elif 'volatility_60d' in X.columns:
            risk = X['volatility_60d'] * 0.5
        else:
            risk = pd.Series([0.15] * len(X), index=X.index)
        
        return risk
    
    def prepare_training_data(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from historical stock data.
        
        Returns:
            Tuple of (features DataFrame, targets Series)
        """
        # Get benchmark data
        benchmark_ticker = self.config.benchmark_ticker
        
        try:
            benchmark_df = self.market_data.get_ohlcv(
                benchmark_ticker,
                start_date,
                end_date
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch benchmark data for {benchmark_ticker}: {e}")
            print(f"⚠️  Continuing without benchmark (returns will be absolute, not excess)")
            benchmark_df = pd.DataFrame()
        
        if benchmark_df.empty:
            print(f"⚠️  Warning: No benchmark data available. Using zero returns (absolute returns instead of excess)")
            # Create empty benchmark returns dict - will result in absolute returns
            benchmark_returns = {}
        else:
            # Compute benchmark returns
            benchmark_df = benchmark_df.sort_values('date')
            benchmark_df['return'] = benchmark_df['close'].pct_change()
            benchmark_returns = dict(zip(benchmark_df['date'], benchmark_df['return']))
        
        # Compute benchmark returns
        benchmark_df = benchmark_df.sort_values('date')
        benchmark_df['return'] = benchmark_df['close'].pct_change()
        benchmark_returns = dict(zip(benchmark_df['date'], benchmark_df['return']))
        
        features_list = []
        targets_list = []
        
        # Iterate through dates and tickers
        current_date = start_date
        lookback_days = self.config.get('training.feature_lookback_days', 252)
        
        # Process monthly instead of weekly to reduce API calls for free tier
        date_increment_days = 30  # Monthly increments
        
        total_combinations = len(tickers) * len(pd.date_range(start_date, end_date, freq=f'{date_increment_days}D'))
        processed = 0
        errors = 0
        
        while current_date <= end_date:
            # Only process weekdays
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            future_date = current_date + timedelta(days=self.target_horizon_days)
            
            # Skip if future date is beyond end_date
            if future_date > end_date:
                current_date += timedelta(days=date_increment_days)
                continue
            
            print(f"Processing date: {current_date} ({processed}/{total_combinations} processed, {errors} errors)")
            
            for ticker in tickers:
                try:
                    # Compute features as of current_date
                    features = self.feature_engine.compute_features(
                        ticker,
                        current_date,
                        lookback_days
                    )
                    
                    # Get future return
                    price_df = self.market_data.get_ohlcv(ticker, current_date, future_date)
                    if price_df.empty:
                        errors += 1
                        continue
                    
                    price_df = price_df.sort_values('date')
                    stock_return = (price_df.iloc[-1]['close'] / price_df.iloc[0]['close']) - 1
                    
                    # Get benchmark return for same period (if available)
                    if benchmark_returns:
                        benchmark_period_return = 0.0
                        for check_date in pd.date_range(current_date, future_date, freq='D'):
                            if check_date.date() in benchmark_returns:
                                benchmark_period_return = (1 + benchmark_period_return) * (1 + benchmark_returns[check_date.date()]) - 1
                        excess_return = stock_return - benchmark_period_return
                    else:
                        # No benchmark available, use absolute return
                        excess_return = stock_return
                    
                    # Store
                    features['ticker'] = ticker
                    features['date'] = current_date
                    features_list.append(features)
                    targets_list.append(excess_return)
                    processed += 1
                    
                    # Small delay between tickers to avoid rate limits
                    time.sleep(0.5)
                    
                except ValueError as e:
                    # Handle API errors gracefully
                    if "403" in str(e) or "Forbidden" in str(e):
                        print(f"  ⚠️  {ticker}: Access denied - may need paid tier for this data")
                        errors += 1
                        time.sleep(2)  # Wait longer after access denied
                    elif "429" in str(e) or "rate limit" in str(e).lower():
                        print(f"  ⚠️  {ticker}: Rate limited - waiting 60 seconds...")
                        time.sleep(60)
                        errors += 1
                    else:
                        print(f"  ⚠️  {ticker} on {current_date}: {e}")
                        errors += 1
                        continue
                except Exception as e:
                    print(f"  ⚠️  Error processing {ticker} on {current_date}: {e}")
                    errors += 1
                    continue
            
            # Move to next date (monthly for efficiency and to reduce API calls)
            current_date += timedelta(days=date_increment_days)
            
            # Longer delay between dates to respect rate limits
            if processed > 0:
                time.sleep(2)
        
        if not features_list:
            raise ValueError("No training data generated")
        
        X = pd.DataFrame(features_list)
        y = pd.Series(targets_list)
        
        # Drop non-feature columns
        X = X.drop(columns=['ticker', 'date'], errors='ignore')
        
        return X, y

