"""Base classes and interfaces for data providers."""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import pandas as pd


class DataProvider(ABC):
    """Base class for all data providers."""
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the provider is accessible."""
        pass


class MarketDataProvider(DataProvider):
    """Abstract interface for market data providers."""
    
    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "day"
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            interval: 'day', 'hour', 'minute', etc.
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Get fundamental data for a stock.
        
        Returns:
            Dictionary with keys like: earnings, revenue, margins, pe_ratio, etc.
        """
        pass
    
    @abstractmethod
    def get_multiple_ohlcv(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        interval: str = "day"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple tickers efficiently.
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        pass
    
    def health_check(self) -> bool:
        """Default health check - can be overridden."""
        try:
            # Try to get data for a common ticker
            end = date.today()
            start = date(end.year, end.month, 1)
            _ = self.get_ohlcv("SPY", start, end)
            return True
        except Exception:
            return False


class NewsProvider(DataProvider):
    """Abstract interface for news providers."""
    
    @abstractmethod
    def get_news(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get news articles.
        
        Args:
            ticker: Optional ticker to filter by
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of articles to return
        
        Returns:
            DataFrame with columns: timestamp, headline, content, tickers, sentiment, category
        """
        pass
    
    @abstractmethod
    def get_news_mentions(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Get time series of news mention counts and sentiment.
        
        Returns:
            DataFrame with columns: date, mention_count, avg_sentiment, max_sentiment, min_sentiment
        """
        pass
    
    def health_check(self) -> bool:
        """Default health check."""
        try:
            _ = self.get_news(limit=1)
            return True
        except Exception:
            return False

