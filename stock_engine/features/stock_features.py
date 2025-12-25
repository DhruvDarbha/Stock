"""Stock-level feature engineering."""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Any

from ..data import get_market_data_provider, get_news_provider
from ..config import get_config


class StockFeatureEngine:
    """Engine for computing stock-level features."""
    
    def __init__(self):
        """Initialize feature engine with data providers."""
        self.market_data = get_market_data_provider()
        self.news_data = get_news_provider()
        self.config = get_config()
    
    def compute_features(
        self,
        ticker: str,
        as_of_date: date,
        lookback_days: int = 252
    ) -> Dict[str, float]:
        """
        Compute all features for a stock as of a given date.
        
        Args:
            ticker: Stock ticker
            as_of_date: Date to compute features as of
            lookback_days: Number of days to look back for historical data
        
        Returns:
            Dictionary of feature name -> value
        """
        start_date = as_of_date - timedelta(days=lookback_days)
        
        # Get price data
        price_df = self.market_data.get_ohlcv(ticker, start_date, as_of_date)
        
        if price_df.empty:
            return self._empty_features()
        
        # Get fundamentals
        fundamentals = self.market_data.get_fundamentals(ticker)
        
        # Get news/sentiment data (handle failures gracefully)
        news_start = as_of_date - timedelta(days=self.config.get('events.sentiment_window_medium_days', 30))
        try:
            news_df = self.news_data.get_news(ticker=ticker, start_date=news_start, end_date=as_of_date)
            mentions_df = self.news_data.get_news_mentions(ticker, news_start, as_of_date)
        except Exception as e:
            # News API failures shouldn't stop feature computation
            news_df = pd.DataFrame()
            mentions_df = pd.DataFrame()
        
        features = {}
        
        # Technical features
        features.update(self._compute_technical_features(price_df))
        
        # Fundamental features
        features.update(self._compute_fundamental_features(fundamentals, price_df))
        
        # Sentiment features
        features.update(self._compute_sentiment_features(news_df, mentions_df, as_of_date))
        
        # Event features
        features.update(self._compute_event_features(news_df, mentions_df, as_of_date))
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary with NaN values."""
        # Return default feature structure with NaN
        return {
            # Technical
            'return_1d': np.nan, 'return_5d': np.nan, 'return_1m': np.nan,
            'return_3m': np.nan, 'return_12m': np.nan,
            'ma20': np.nan, 'ma50': np.nan, 'ma200': np.nan,
            'distance_52w_high': np.nan, 'distance_52w_low': np.nan,
            'volatility_20d': np.nan, 'volatility_60d': np.nan,
            'atr_14': np.nan, 'avg_volume': np.nan,
            # Fundamental (would need sector data for relative metrics)
            'pe_ratio': np.nan, 'ev_ebitda': np.nan, 'ps_ratio': np.nan,
            'revenue_growth_yoy': np.nan, 'earnings_growth_yoy': np.nan,
            'gross_margin': np.nan, 'net_margin': np.nan,
            'roe': np.nan, 'roic': np.nan, 'debt_to_equity': np.nan,
            # Sentiment
            'sentiment_7d': np.nan, 'sentiment_30d': np.nan,
            'mention_count_7d': 0.0, 'mention_count_30d': 0.0,
            # Events
            'shock_indicator': 0.0, 'recent_negative_event': 0.0,
            'recent_positive_event': 0.0
        }
    
    def _compute_technical_features(self, price_df: pd.DataFrame) -> Dict[str, float]:
        """Compute technical/price-based features."""
        if price_df.empty:
            return {}
        
        # Ensure sorted by date
        price_df = price_df.sort_values('date').reset_index(drop=True)
        
        features = {}
        latest = price_df.iloc[-1]
        
        # Returns
        if len(price_df) >= 1:
            features['return_1d'] = (latest['close'] / price_df.iloc[-2]['close'] - 1) if len(price_df) >= 2 else 0.0
        
        if len(price_df) >= 5:
            features['return_5d'] = (latest['close'] / price_df.iloc[-6]['close'] - 1) if len(price_df) >= 6 else 0.0
        
        if len(price_df) >= 20:
            features['return_1m'] = (latest['close'] / price_df.iloc[-21]['close'] - 1)
        
        if len(price_df) >= 63:
            features['return_3m'] = (latest['close'] / price_df.iloc[-64]['close'] - 1)
        
        if len(price_df) >= 252:
            features['return_12m'] = (latest['close'] / price_df.iloc[-253]['close'] - 1)
        
        # Moving averages
        if len(price_df) >= 20:
            features['ma20'] = price_df['close'].tail(20).mean()
            features['distance_from_ma20'] = (latest['close'] / features['ma20'] - 1)
        
        if len(price_df) >= 50:
            features['ma50'] = price_df['close'].tail(50).mean()
            features['distance_from_ma50'] = (latest['close'] / features['ma50'] - 1)
        
        if len(price_df) >= 200:
            features['ma200'] = price_df['close'].tail(200).mean()
            features['distance_from_ma200'] = (latest['close'] / features['ma200'] - 1)
        
        # 52-week high/low
        if len(price_df) >= 252:
            high_52w = price_df['high'].tail(252).max()
            low_52w = price_df['low'].tail(252).min()
            features['distance_52w_high'] = (latest['close'] / high_52w - 1)
            features['distance_52w_low'] = (latest['close'] / low_52w - 1)
        
        # Volatility
        if len(price_df) >= 20:
            returns_20d = price_df['close'].tail(20).pct_change().dropna()
            features['volatility_20d'] = returns_20d.std() * np.sqrt(252)  # Annualized
        
        if len(price_df) >= 60:
            returns_60d = price_df['close'].tail(60).pct_change().dropna()
            features['volatility_60d'] = returns_60d.std() * np.sqrt(252)  # Annualized
        
        # ATR (Average True Range)
        if len(price_df) >= 14:
            high_low = price_df['high'] - price_df['low']
            high_close = (price_df['high'] - price_df['close'].shift(1)).abs()
            low_close = (price_df['low'] - price_df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr_14'] = tr.tail(14).mean()
        
        # Volume
        features['avg_volume'] = price_df['volume'].tail(20).mean() if len(price_df) >= 20 else price_df['volume'].mean()
        features['volume_ratio'] = latest['volume'] / features['avg_volume'] if features['avg_volume'] > 0 else 1.0
        
        return features
    
    def _compute_fundamental_features(
        self,
        fundamentals: Dict[str, Any],
        price_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute fundamental-based features."""
        features = {}
        
        if not fundamentals or price_df.empty:
            return {}
        
        # Direct fundamental metrics
        features['pe_ratio'] = fundamentals.get('pe_ratio', np.nan)
        features['ev_ebitda'] = fundamentals.get('ev_ebitda', np.nan)
        features['ps_ratio'] = fundamentals.get('ps_ratio', np.nan)
        features['revenue_growth_yoy'] = fundamentals.get('revenue_growth_yoy', np.nan)
        features['earnings_growth_yoy'] = fundamentals.get('earnings_growth_yoy', np.nan)
        features['gross_margin'] = fundamentals.get('gross_margin', np.nan)
        features['net_margin'] = fundamentals.get('net_margin', np.nan)
        features['roe'] = fundamentals.get('roe', np.nan)
        features['roic'] = fundamentals.get('roic', np.nan)
        features['debt_to_equity'] = fundamentals.get('debt_to_equity', np.nan)
        
        # TODO: Add relative metrics (vs sector median) when sector data available
        
        return features
    
    def _compute_sentiment_features(
        self,
        news_df: pd.DataFrame,
        mentions_df: pd.DataFrame,
        as_of_date: date
    ) -> Dict[str, float]:
        """Compute sentiment-based features."""
        features = {}
        
        if news_df.empty or 'timestamp' not in news_df.columns:
            features['sentiment_7d'] = 0.0
            features['sentiment_30d'] = 0.0
            features['mention_count_7d'] = 0.0
            features['mention_count_30d'] = 0.0
            return features
        
        try:
            # Convert sentiment to numeric
            sentiment_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            news_df['sentiment_numeric'] = news_df['sentiment'].map(sentiment_map)
            
            # Filter by date windows
            news_df['date'] = pd.to_datetime(news_df['timestamp']).dt.date
        
            date_7d = as_of_date - timedelta(days=7)
            date_30d = as_of_date - timedelta(days=30)
            
            news_7d = news_df[news_df['date'] >= date_7d]
            news_30d = news_df[news_df['date'] >= date_30d]
            
            # Average sentiment
            features['sentiment_7d'] = news_7d['sentiment_numeric'].mean() if not news_7d.empty and 'sentiment_numeric' in news_7d.columns else 0.0
            features['sentiment_30d'] = news_30d['sentiment_numeric'].mean() if not news_30d.empty and 'sentiment_numeric' in news_30d.columns else 0.0
            
            # Mention counts
            features['mention_count_7d'] = len(news_7d)
            features['mention_count_30d'] = len(news_30d)
        except Exception as e:
            # If anything fails, return zeros
            features['sentiment_7d'] = 0.0
            features['sentiment_30d'] = 0.0
            features['mention_count_7d'] = 0.0
            features['mention_count_30d'] = 0.0
        
        return features
    
    def _compute_event_features(
        self,
        news_df: pd.DataFrame,
        mentions_df: pd.DataFrame,
        as_of_date: date
    ) -> Dict[str, float]:
        """Compute event/shock indicators."""
        features = {}
        
        try:
            if mentions_df.empty or 'mention_count' not in mentions_df.columns:
                features['shock_indicator'] = 0.0
                features['recent_negative_event'] = 0.0
                features['recent_positive_event'] = 0.0
                return features
            
            # Shock indicator: z-score of recent mention volume vs historical
            if len(mentions_df) >= 30:
                recent_mentions = mentions_df['mention_count'].tail(7).mean()
                historical_mentions = mentions_df['mention_count'].head(len(mentions_df) - 7).mean()
                historical_std = mentions_df['mention_count'].head(len(mentions_df) - 7).std()
                
                if historical_std > 0:
                    features['shock_indicator'] = (recent_mentions - historical_mentions) / historical_std
                else:
                    features['shock_indicator'] = 0.0
            else:
                features['shock_indicator'] = 0.0
            
            # Recent negative/positive events
            if not news_df.empty and 'timestamp' in news_df.columns:
                date_7d = as_of_date - timedelta(days=7)
                news_df_copy = news_df.copy()
                news_df_copy['date'] = pd.to_datetime(news_df_copy['timestamp']).dt.date
                recent_news = news_df_copy[news_df_copy['date'] >= date_7d]
                
                if not recent_news.empty and 'sentiment' in recent_news.columns:
                    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                    recent_news = recent_news.copy()  # Fix SettingWithCopyWarning
                    recent_news['sentiment_numeric'] = recent_news['sentiment'].map(sentiment_map)
                    
                    # Count extreme events
                    features['recent_negative_event'] = (recent_news['sentiment_numeric'] < -0.5).sum()
                    features['recent_positive_event'] = (recent_news['sentiment_numeric'] > 0.5).sum()
                else:
                    features['recent_negative_event'] = 0.0
                    features['recent_positive_event'] = 0.0
            else:
                features['recent_negative_event'] = 0.0
                features['recent_positive_event'] = 0.0
        except Exception as e:
            # Fallback on any error
            features['shock_indicator'] = 0.0
            features['recent_negative_event'] = 0.0
            features['recent_positive_event'] = 0.0
        
        return features
    
    def compute_features_batch(
        self,
        tickers: List[str],
        as_of_date: date,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Compute features for multiple tickers.
        
        Returns:
            DataFrame with ticker as index and features as columns
        """
        results = []
        
        for ticker in tickers:
            try:
                features = self.compute_features(ticker, as_of_date, lookback_days)
                features['ticker'] = ticker
                results.append(features)
            except Exception as e:
                print(f"Error computing features for {ticker}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df.set_index('ticker', inplace=True)
        return df

