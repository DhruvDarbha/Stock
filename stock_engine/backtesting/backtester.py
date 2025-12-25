"""Backtesting framework for the recommendation engine."""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from ..models import MarketModel
from ..portfolio import PortfolioOptimizer, TradeGenerator
from ..data import get_market_data_provider, get_user_data_store
from ..config import get_config


class Backtester:
    """Backtest portfolio recommendations over historical period."""
    
    def __init__(self, market_model: MarketModel):
        """
        Initialize backtester.
        
        Args:
            market_model: Trained market model
        """
        self.market_model = market_model
        self.market_data = get_market_data_provider()
        self.config = get_config()
        self.benchmark_ticker = self.config.benchmark_ticker
    
    def run_backtest(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100000.0,
        rebalance_frequency_days: int = 30,
        use_behavior_model: bool = False
    ) -> pd.DataFrame:
        """
        Run backtest over historical period.
        
        Args:
            tickers: List of tickers to trade
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            rebalance_frequency_days: How often to rebalance
            use_behavior_model: Whether to use behavior model (if available)
        
        Returns:
            DataFrame with columns: date, portfolio_value, benchmark_value, returns, benchmark_returns
        """
        # Initialize portfolio
        portfolio_value = initial_capital
        positions = {}  # ticker -> shares
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(self.market_model)
        trade_generator = TradeGenerator(optimizer)
        
        # Get benchmark data
        benchmark_df = self.market_data.get_ohlcv(
            self.benchmark_ticker,
            start_date,
            end_date
        )
        benchmark_df = benchmark_df.sort_values('date')
        benchmark_df['price'] = benchmark_df['close']
        benchmark_df = benchmark_df.set_index('date')
        
        # Track results
        results = []
        current_date = start_date
        last_rebalance_date = start_date
        
        print(f"Running backtest from {start_date} to {end_date}")
        print(f"Initial capital: ${initial_capital:,.2f}")
        print(f"Rebalance frequency: {rebalance_frequency_days} days")
        
        while current_date <= end_date:
            # Check if we should rebalance
            days_since_rebalance = (current_date - last_rebalance_date).days
            
            if days_since_rebalance >= rebalance_frequency_days:
                # Rebalance
                print(f"\nRebalancing on {current_date}...")
                
                # Compute current weights
                current_weights = {}
                total_value = 0.0
                current_prices = {}
                
                for ticker in tickers:
                    try:
                        price_df = self.market_data.get_ohlcv(ticker, current_date, current_date)
                        if not price_df.empty:
                            price = price_df.iloc[-1]['close']
                            current_prices[ticker] = price
                            shares = positions.get(ticker, 0)
                            value = shares * price
                            total_value += value
                    except:
                        continue
                
                if total_value == 0:
                    total_value = portfolio_value
                
                for ticker in tickers:
                    if ticker in positions and ticker in current_prices:
                        value = positions[ticker] * current_prices[ticker]
                        current_weights[ticker] = value / total_value
                    else:
                        current_weights[ticker] = 0.0
                
                # Get target weights
                target_weights_df = optimizer.compute_target_weights(
                    tickers,
                    current_weights,
                    as_of_date=current_date
                )
                
                if not target_weights_df.empty:
                    # Execute trades (simplified - no transaction costs for now)
                    target_weights = dict(zip(
                        target_weights_df['ticker'],
                        target_weights_df['target_weight']
                    ))
                    
                    # Rebalance positions
                    for ticker in tickers:
                        target_weight = target_weights.get(ticker, 0.0)
                        current_price = current_prices.get(ticker)
                        
                        if current_price:
                            target_value = target_weight * total_value
                            target_shares = int(target_value / current_price)
                            positions[ticker] = target_shares
                    
                    portfolio_value = total_value
                    last_rebalance_date = current_date
            
            # Calculate current portfolio value
            current_portfolio_value = 0.0
            for ticker, shares in positions.items():
                try:
                    price_df = self.market_data.get_ohlcv(ticker, current_date, current_date)
                    if not price_df.empty:
                        price = price_df.iloc[-1]['close']
                        current_portfolio_value += shares * price
                except:
                    continue
            
            # Get benchmark value
            if current_date in benchmark_df.index:
                benchmark_price = benchmark_df.loc[current_date, 'price']
                if 'benchmark_shares' not in locals():
                    benchmark_shares = initial_capital / benchmark_price
                benchmark_value = benchmark_shares * benchmark_price
            else:
                benchmark_value = current_portfolio_value  # Fallback
            
            # Calculate returns
            portfolio_return = (current_portfolio_value / initial_capital) - 1
            benchmark_return = (benchmark_value / initial_capital) - 1
            
            results.append({
                'date': current_date,
                'portfolio_value': current_portfolio_value,
                'benchmark_value': benchmark_value,
                'portfolio_return': portfolio_return,
                'benchmark_return': benchmark_return
            })
            
            # Move to next date (weekly for efficiency)
            current_date += timedelta(days=7)
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def compute_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute backtest performance metrics.
        
        Args:
            results_df: DataFrame from run_backtest
        
        Returns:
            Dictionary of metrics
        """
        if results_df.empty:
            return {}
        
        # Annualized returns
        total_days = (results_df['date'].max() - results_df['date'].min()).days
        years = total_days / 365.25
        
        portfolio_final_return = results_df['portfolio_return'].iloc[-1]
        benchmark_final_return = results_df['benchmark_return'].iloc[-1]
        
        portfolio_cagr = (1 + portfolio_final_return) ** (1 / years) - 1 if years > 0 else 0
        benchmark_cagr = (1 + benchmark_final_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Excess return
        excess_return = portfolio_cagr - benchmark_cagr
        
        # Volatility (annualized)
        portfolio_returns = results_df['portfolio_return'].diff().dropna()
        benchmark_returns = results_df['benchmark_return'].diff().dropna()
        
        portfolio_vol = portfolio_returns.std() * np.sqrt(252 / 7)  # Weekly data
        benchmark_vol = benchmark_returns.std() * np.sqrt(252 / 7)
        
        # Sharpe ratio (assume risk-free rate = 0)
        portfolio_sharpe = portfolio_cagr / portfolio_vol if portfolio_vol > 0 else 0
        benchmark_sharpe = benchmark_cagr / benchmark_vol if benchmark_vol > 0 else 0
        
        # Max drawdown
        portfolio_cummax = results_df['portfolio_value'].cummax()
        portfolio_drawdown = (results_df['portfolio_value'] - portfolio_cummax) / portfolio_cummax
        max_drawdown = portfolio_drawdown.min()
        
        benchmark_cummax = results_df['benchmark_value'].cummax()
        benchmark_drawdown = (results_df['benchmark_value'] - benchmark_cummax) / benchmark_cummax
        benchmark_max_drawdown = benchmark_drawdown.min()
        
        return {
            'portfolio_cagr': portfolio_cagr,
            'benchmark_cagr': benchmark_cagr,
            'excess_return': excess_return,
            'portfolio_volatility': portfolio_vol,
            'benchmark_volatility': benchmark_vol,
            'portfolio_sharpe': portfolio_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'total_return': portfolio_final_return,
            'benchmark_total_return': benchmark_final_return
        }

