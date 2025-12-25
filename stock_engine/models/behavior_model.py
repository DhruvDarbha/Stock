"""User behavior prediction model."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from .base_model import BaseModel
from ..features import UserFeatureEngine
from ..data import get_user_data_store
from ..config import get_config


class BehaviorModel(BaseModel):
    """Model for predicting user behavior (follow probability, panic exits)."""
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Initialize behavior model.
        
        Args:
            model_type: 'lightgbm', 'xgboost', or 'catboost'
        """
        super().__init__(model_type)
        self.feature_engine = UserFeatureEngine()
        self.user_store = get_user_data_store()
        self.config = get_config()
        self.follow_threshold = self.config.get('models.behavior_model.follow_threshold', 0.5)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        task: str = 'follow',
        **kwargs
    ):
        """
        Train the behavior model.
        
        Args:
            X: Feature matrix (user features + recommendation features)
            y: Target vector (1 if followed, 0 if not, for 'follow' task)
            validation_split: Fraction to use for validation
            task: 'follow' (binary classification) or 'panic' (binary classification)
            **kwargs: Additional training parameters
        """
        self.task = task
        
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
        """Train LightGBM classifier."""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
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
        """Train XGBoost classifier."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0
        }
        params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**params)
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
        """Train CatBoost classifier."""
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'verbose': False,
            'early_stopping_rounds': 50
        }
        params.update(kwargs)
        
        self.model = CatBoostClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
    
    def predict_follow_probability(
        self,
        user_id: int,
        recommendation_features: Dict[str, float]
    ) -> float:
        """
        Predict probability that user will follow a recommendation.
        
        Args:
            user_id: User ID
            recommendation_features: Features of the recommendation (ticker, size, risk, etc.)
        
        Returns:
            Probability (0-1) that user will follow
        """
        if not self.is_trained or self.task != 'follow':
            raise ValueError("Model must be trained on 'follow' task")
        
        # Get user features
        user_features = self.feature_engine.compute_features(user_id)
        
        # Combine user and recommendation features
        combined_features = {**user_features, **recommendation_features}
        
        # Convert to DataFrame
        X = pd.DataFrame([combined_features])
        
        # Ensure feature order matches training
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                # Fill missing with 0
                for feat in missing_features:
                    X[feat] = 0
            X = X[self.feature_names]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Predict
        if self.model_type == 'lightgbm':
            prob = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
        else:
            prob = self.model.predict_proba(X)[0, 1]
        
        return float(prob)
    
    def predict_panic_probability(
        self,
        user_id: int,
        drawdown: float,
        volatility: float
    ) -> float:
        """
        Predict probability of panic exit given drawdown and volatility.
        
        Args:
            user_id: User ID
            drawdown: Current drawdown (negative number)
            volatility: Current volatility
        
        Returns:
            Probability (0-1) of panic exit
        """
        if not self.is_trained or self.task != 'panic':
            raise ValueError("Model must be trained on 'panic' task")
        
        # Get user features
        user_features = self.feature_engine.compute_features(user_id)
        
        # Add drawdown and volatility
        features = {
            **user_features,
            'current_drawdown': abs(drawdown),
            'current_volatility': volatility
        }
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Ensure feature order matches training
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                for feat in missing_features:
                    X[feat] = 0
            X = X[self.feature_names]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Predict
        if self.model_type == 'lightgbm':
            prob = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
        else:
            prob = self.model.predict_proba(X)[0, 1]
        
        return float(prob)
    
    def prepare_training_data(
        self,
        user_ids: List[int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        task: str = 'follow'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from user recommendation and action history.
        
        Args:
            user_ids: List of user IDs
            start_date: Start date for data
            end_date: End date for data
            task: 'follow' or 'panic'
        
        Returns:
            Tuple of (features DataFrame, targets Series)
        """
        features_list = []
        targets_list = []
        
        for user_id in user_ids:
            try:
                user_features = self.feature_engine.compute_features(user_id)
                
                if task == 'follow':
                    # Get recommendations and corresponding actions
                    recommendations = self.user_store.get_recommendations(
                        user_id,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if recommendations.empty:
                        continue
                    
                    for _, rec in recommendations.iterrows():
                        # Get corresponding action
                        actions = self.user_store.get_user_actions(user_id)
                        matched_actions = actions[actions['source_recommendation_id'] == rec['id']]
                        
                        if matched_actions.empty:
                            # No action taken
                            followed = 0
                        else:
                            # Check if action matches recommendation
                            action = matched_actions.iloc[0]
                            if action['action_taken'] == rec['action']:
                                # Check if size is close (within threshold)
                                if rec['shares'] and action['shares']:
                                    follow_ratio = action['shares'] / rec['shares']
                                    followed = 1 if follow_ratio >= self.follow_threshold else 0
                                else:
                                    followed = 1
                            else:
                                followed = 0
                        
                        # Recommendation features
                        rec_features = {
                            'recommended_action_buy': 1.0 if rec['action'] == 'buy' else 0.0,
                            'recommended_action_sell': 1.0 if rec['action'] == 'sell' else 0.0,
                            'recommended_shares': rec.get('shares', 0.0) or 0.0,
                            'recommended_weight': rec.get('weight', 0.0) or 0.0
                        }
                        
                        combined = {**user_features, **rec_features}
                        features_list.append(combined)
                        targets_list.append(followed)
                
                elif task == 'panic':
                    # TODO: Implement panic exit detection from position history
                    # This would require tracking portfolio value over time
                    # For now, skip
                    continue
                    
            except Exception as e:
                print(f"Error preparing data for user {user_id}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No training data generated")
        
        X = pd.DataFrame(features_list)
        y = pd.Series(targets_list)
        
        return X, y

