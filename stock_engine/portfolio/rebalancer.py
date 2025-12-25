"""Trade generation from portfolio optimization results."""

import pandas as pd
from typing import List, Dict, Optional
from datetime import date, datetime, timedelta

from .optimizer import PortfolioOptimizer
from ..data import get_user_data_store, get_market_data_provider
from ..config import get_config


class TradeGenerator:
    """Generate concrete trade recommendations from target weights."""
    
    def __init__(self, optimizer: PortfolioOptimizer):
        """
        Initialize trade generator.
        
        Args:
            optimizer: PortfolioOptimizer instance
        """
        self.optimizer = optimizer
        self.user_store = get_user_data_store()
        self.market_data = get_market_data_provider()
        self.config = get_config()
    
    def generate_trades(
        self,
        user_id: int,
        tickers: List[str],
        portfolio_value: float,
        as_of_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Generate trade recommendations for a user.
        
        Args:
            user_id: User ID
            tickers: List of tickers to consider
            portfolio_value: Total portfolio value in dollars
            as_of_date: Date for recommendations
        
        Returns:
            DataFrame with columns: ticker, action, shares, current_weight, target_weight, rationale
        """
        as_of_date = as_of_date or date.today()
        
        # Get current positions
        positions_df = self.user_store.get_positions(user_id)
        
        # Convert to weight dictionary
        current_positions = {}
        if not positions_df.empty:
            # Get current prices to compute current weights
            for _, pos in positions_df.iterrows():
                ticker = pos['ticker']
                shares = pos['shares']
                
            # Get current price
            try:
                # Try to get recent price (last 5 days in case as_of_date is weekend)
                price_df = self.market_data.get_ohlcv(
                    ticker,
                    as_of_date - timedelta(days=5),
                    as_of_date
                )
                if not price_df.empty:
                    current_price = price_df.iloc[-1]['close']
                    current_value = shares * current_price
                    current_positions[ticker] = current_value / portfolio_value
                else:
                    current_positions[ticker] = 0.0
            except Exception as e:
                current_positions[ticker] = 0.0
        
        # Get target weights
        target_weights_df = self.optimizer.compute_target_weights(
            tickers,
            current_positions,
            user_id=user_id,
            as_of_date=as_of_date
        )
        
        if target_weights_df.empty:
            return pd.DataFrame()
        
        # Get current prices for all tickers
        current_prices = {}
        for ticker in tickers:
            try:
                # Try to get recent price (last 5 days in case as_of_date is weekend)
                price_df = self.market_data.get_ohlcv(
                    ticker,
                    as_of_date - timedelta(days=5),
                    as_of_date
                )
                if not price_df.empty:
                    current_prices[ticker] = price_df.iloc[-1]['close']
                else:
                    current_prices[ticker] = None
            except Exception as e:
                current_prices[ticker] = None
        
        # Generate trades
        trades = []
        rebalance_threshold = self.config.get('portfolio.rebalance_threshold', 0.01)
        max_turnover = self.config.get('portfolio.max_turnover_per_rebalance', 0.20)
        
        total_turnover = 0.0
        
        for _, row in target_weights_df.iterrows():
            ticker = row['ticker']
            current_weight = row['current_weight']
            target_weight = row['target_weight']
            current_price = current_prices.get(ticker)
            
            if current_price is None:
                continue
            
            weight_diff = target_weight - current_weight
            
            # Only generate trade if difference exceeds threshold
            if abs(weight_diff) < rebalance_threshold:
                trades.append({
                    'ticker': ticker,
                    'action': 'hold',
                    'shares': 0,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'rationale': 'Within rebalance threshold'
                })
                continue
            
            # Compute target dollar amount
            target_value = target_weight * portfolio_value
            current_value = current_weight * portfolio_value
            
            # Compute shares to trade
            value_diff = target_value - current_value
            shares_diff = value_diff / current_price
            
            # Round shares
            if abs(shares_diff) < 0.01:
                action = 'hold'
                shares = 0
            elif shares_diff > 0:
                action = 'buy'
                shares = int(shares_diff)
            else:
                action = 'sell'
                shares = abs(int(shares_diff))
            
            # Check turnover constraint
            trade_turnover = abs(shares * current_price) / portfolio_value
            if total_turnover + trade_turnover > max_turnover:
                # Scale down trade
                remaining_turnover = max_turnover - total_turnover
                max_trade_value = remaining_turnover * portfolio_value
                shares = int(max_trade_value / current_price)
                if shares == 0:
                    action = 'hold'
            
            if action != 'hold':
                total_turnover += abs(shares * current_price) / portfolio_value
            
            # Generate rationale
            rationale = self._generate_rationale(row, action, shares)
            
            # Calculate dollar amounts
            if action == 'buy':
                investment_amount = shares * current_price
            elif action == 'sell':
                investment_amount = -shares * current_price  # Negative for sell
            else:
                investment_amount = 0.0
            
            trades.append({
                'ticker': ticker,
                'action': action,
                'shares': shares,
                'current_price': current_price,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'expected_excess_return': row.get('expected_excess_return', 0.0),
                'risk_score': row.get('risk_score', 0.15),
                'investment_amount': investment_amount,
                'rationale': rationale
            })
        
        return pd.DataFrame(trades)
    
    def _generate_rationale(
        self,
        row: pd.Series,
        action: str,
        shares: int
    ) -> str:
        """Generate human-readable rationale for a trade."""
        ticker = row['ticker']
        expected_return = row.get('expected_excess_return', 0.0)
        risk_score = row.get('risk_score', 0.15)
        
        rationale_parts = []
        
        if action == 'buy':
            rationale_parts.append(f"Buy {shares} shares of {ticker}")
            if expected_return > 0:
                rationale_parts.append(f"Expected excess return: {expected_return:.2%}")
            if risk_score > 0:
                rationale_parts.append(f"Risk score: {risk_score:.2f}")
        elif action == 'sell':
            rationale_parts.append(f"Sell {shares} shares of {ticker}")
            if expected_return < 0:
                rationale_parts.append(f"Negative expected excess return: {expected_return:.2%}")
            rationale_parts.append("Rebalancing to target weights")
        else:
            rationale_parts.append(f"Hold {ticker} - within rebalance threshold")
        
        return ". ".join(rationale_parts)
    
    def save_recommendations(
        self,
        user_id: int,
        trades_df: pd.DataFrame,
        model_version: str = "v1.0"
    ):
        """
        Save trade recommendations to database.
        
        Args:
            user_id: User ID
            trades_df: DataFrame with trade recommendations
            model_version: Model version string
        """
        for _, row in trades_df.iterrows():
            self.user_store.create_recommendation(
                user_id=user_id,
                ticker=row['ticker'],
                action=row['action'],
                shares=row['shares'] if row['action'] != 'hold' else None,
                weight=row['target_weight'],
                rationale=row.get('rationale', ''),
                model_version=model_version
            )

