"""News data provider implementations."""

import pandas as pd
import requests
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
import time
import numpy as np

from .base import NewsProvider
from ..config import get_config


class MockNewsProvider(NewsProvider):
    """Mock news provider for testing and development."""
    
    def __init__(self):
        """Initialize mock provider."""
        self._news_cache: List[Dict] = []
        self._generate_mock_news()
    
    def _generate_mock_news(self):
        """Generate mock news articles."""
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'KO', 'PEP']
        categories = ['earnings', 'product', 'executive', 'regulatory', 'partnership', 'lawsuit']
        sentiments = ['positive', 'negative', 'neutral']
        
        # Generate news for last 90 days
        base_date = date.today()
        for i in range(90):
            news_date = base_date - timedelta(days=i)
            
            # Generate 0-5 news items per day
            num_news = np.random.poisson(2)
            for _ in range(num_news):
                ticker = np.random.choice(tickers)
                category = np.random.choice(categories)
                sentiment = np.random.choice(sentiments, p=[0.3, 0.2, 0.5])
                
                headlines = {
                    'earnings': f"{ticker} Reports Strong Q4 Earnings",
                    'product': f"{ticker} Launches New Product Line",
                    'executive': f"{ticker} CEO Makes Major Announcement",
                    'regulatory': f"Regulators Investigate {ticker}",
                    'partnership': f"{ticker} Partners with Major Company",
                    'lawsuit': f"{ticker} Faces Class Action Lawsuit"
                }
                
                self._news_cache.append({
                    'timestamp': datetime.combine(news_date, datetime.min.time()),
                    'headline': headlines[category],
                    'content': f"Detailed content about {ticker} and {category}...",
                    'tickers': [ticker],
                    'sentiment': sentiment,
                    'category': category,
                    'source': 'Mock News',
                    'url': f"https://mock.news/{ticker}/{i}"
                })
    
    def get_news(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get mock news articles."""
        filtered = self._news_cache
        
        if ticker:
            filtered = [n for n in filtered if ticker in n.get('tickers', [])]
        
        if start_date:
            filtered = [n for n in filtered if n['timestamp'].date() >= start_date]
        
        if end_date:
            filtered = [n for n in filtered if n['timestamp'].date() <= end_date]
        
        # Sort by timestamp descending
        filtered.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return pd.DataFrame(filtered[:limit])
    
    def get_news_mentions(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Get news mention time series."""
        news_df = self.get_news(ticker=ticker, start_date=start_date, end_date=end_date)
        
        if news_df.empty:
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.DataFrame({
                'date': dates,
                'mention_count': 0,
                'avg_sentiment': 0.0,
                'max_sentiment': 0.0,
                'min_sentiment': 0.0
            })
        
        # Convert sentiment to numeric
        sentiment_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
        news_df['sentiment_numeric'] = news_df['sentiment'].map(sentiment_map)
        
        # Group by date
        news_df['date'] = pd.to_datetime(news_df['timestamp']).dt.date
        daily = news_df.groupby('date').agg({
            'sentiment_numeric': ['count', 'mean', 'max', 'min']
        }).reset_index()
        
        daily.columns = ['date', 'mention_count', 'avg_sentiment', 'max_sentiment', 'min_sentiment']
        
        # Fill missing dates with zeros
        all_dates = pd.date_range(start_date, end_date, freq='D').date
        all_dates_df = pd.DataFrame({'date': all_dates})
        daily = all_dates_df.merge(daily, on='date', how='left').fillna(0)
        
        return daily


class FinnhubNewsProvider(NewsProvider):
    """Finnhub news provider."""
    
    def __init__(self, api_key: str, base_url: str = "https://finnhub.io/api/v1"):
        """
        Initialize Finnhub provider.
        
        Args:
            api_key: Finnhub API key
            base_url: Base URL for API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit_delay = 60.0 / get_config().get('data_sources.news.rate_limit', 60)
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def get_news(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get news from Finnhub."""
        self._rate_limit()
        
        if ticker:
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': ticker,
                'from': start_date.isoformat() if start_date else (date.today() - timedelta(days=30)).isoformat(),
                'to': end_date.isoformat() if end_date else date.today().isoformat(),
                'token': self.api_key
            }
        else:
            url = f"{self.base_url}/news"
            params = {
                'category': 'general',
                'token': self.api_key
            }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json()
        
        if not articles:
            return pd.DataFrame(columns=['timestamp', 'headline', 'content', 'tickers', 'sentiment', 'category'])
        
        # Convert to DataFrame
        data = []
        for article in articles[:limit]:
            # Finnhub doesn't provide sentiment, so we'll need to compute it
            # For now, set as neutral
            data.append({
                'timestamp': pd.to_datetime(article.get('datetime', 0), unit='s'),
                'headline': article.get('headline', ''),
                'content': article.get('summary', ''),
                'tickers': article.get('related', '').split(',') if article.get('related') else [],
                'sentiment': 'neutral',  # TODO: Add sentiment analysis
                'category': article.get('category', 'general'),
                'source': article.get('source', ''),
                'url': article.get('url', '')
            })
        
        return pd.DataFrame(data)
    
    def get_news_mentions(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Get news mention time series from Finnhub."""
        news_df = self.get_news(ticker=ticker, start_date=start_date, end_date=end_date, limit=1000)
        
        if news_df.empty:
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.DataFrame({
                'date': dates,
                'mention_count': 0,
                'avg_sentiment': 0.0,
                'max_sentiment': 0.0,
                'min_sentiment': 0.0
            })
        
        # Convert sentiment to numeric
        sentiment_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
        news_df['sentiment_numeric'] = news_df['sentiment'].map(sentiment_map)
        
        # Group by date
        news_df['date'] = pd.to_datetime(news_df['timestamp']).dt.date
        daily = news_df.groupby('date').agg({
            'sentiment_numeric': ['count', 'mean', 'max', 'min']
        }).reset_index()
        
        daily.columns = ['date', 'mention_count', 'avg_sentiment', 'max_sentiment', 'min_sentiment']
        
        # Fill missing dates
        all_dates = pd.date_range(start_date, end_date, freq='D').date
        all_dates_df = pd.DataFrame({'date': all_dates})
        daily = all_dates_df.merge(daily, on='date', how='left').fillna(0)
        
        return daily
    
    def health_check(self) -> bool:
        """Check if Finnhub API is accessible."""
        try:
            _ = self.get_news(limit=1)
            return True
        except Exception:
            return False


class NewsAPIProvider(NewsProvider):
    """NewsAPI.org news provider."""
    
    def __init__(self, api_key: str, base_url: str = "https://newsapi.org/v2"):
        """
        Initialize NewsAPI provider.
        
        Args:
            api_key: NewsAPI API key
            base_url: Base URL for API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit_delay = 60.0 / get_config().get('data_sources.news.rate_limit', 60)
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def get_news(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get news from NewsAPI."""
        self._rate_limit()
        
        # NewsAPI uses "everything" endpoint for search
        url = f"{self.base_url}/everything"
        params = {
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(limit, 100)  # NewsAPI max is 100 per request
        }
        
        # Add query based on ticker or general finance news
        if ticker:
            # Search for ticker symbol and company name
            params['q'] = ticker
        else:
            # General business/finance news
            params['q'] = 'stock market OR finance OR business'
        
        # Add date filters
        if start_date:
            params['from'] = start_date.isoformat()
        if end_date:
            params['to'] = end_date.isoformat()
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'ok' or not data.get('articles'):
            return pd.DataFrame(columns=['timestamp', 'headline', 'content', 'tickers', 'sentiment', 'category'])
        
        articles = data['articles']
        
        # Convert to DataFrame
        news_data = []
        for article in articles[:limit]:
            # Parse timestamp
            try:
                timestamp = pd.to_datetime(article.get('publishedAt', ''))
            except:
                timestamp = pd.Timestamp.now()
            
            # Extract ticker mentions from title and description
            tickers_mentioned = []
            if ticker:
                text = f"{article.get('title', '')} {article.get('description', '')}".upper()
                if ticker.upper() in text:
                    tickers_mentioned = [ticker]
            
            # Extract content
            content = article.get('description', '') or article.get('content', '') or ''
            
            news_data.append({
                'timestamp': timestamp,
                'headline': article.get('title', ''),
                'content': content[:500] if content else '',  # Truncate long content
                'tickers': tickers_mentioned,
                'sentiment': 'neutral',  # NewsAPI doesn't provide sentiment, will need to compute
                'category': 'general',
                'source': article.get('source', {}).get('name', ''),
                'url': article.get('url', '')
            })
        
        return pd.DataFrame(news_data)
    
    def get_news_mentions(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Get news mention time series from NewsAPI."""
        news_df = self.get_news(ticker=ticker, start_date=start_date, end_date=end_date, limit=1000)
        
        if news_df.empty:
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.DataFrame({
                'date': dates,
                'mention_count': 0,
                'avg_sentiment': 0.0,
                'max_sentiment': 0.0,
                'min_sentiment': 0.0
            })
        
        # Convert sentiment to numeric
        sentiment_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
        news_df['sentiment_numeric'] = news_df['sentiment'].map(sentiment_map)
        
        # Group by date
        news_df['date'] = pd.to_datetime(news_df['timestamp']).dt.date
        daily = news_df.groupby('date').agg({
            'sentiment_numeric': ['count', 'mean', 'max', 'min']
        }).reset_index()
        
        daily.columns = ['date', 'mention_count', 'avg_sentiment', 'max_sentiment', 'min_sentiment']
        
        # Fill missing dates
        all_dates = pd.date_range(start_date, end_date, freq='D').date
        all_dates_df = pd.DataFrame({'date': all_dates})
        daily = all_dates_df.merge(daily, on='date', how='left').fillna(0)
        
        return daily
    
    def health_check(self) -> bool:
        """Check if NewsAPI is accessible."""
        try:
            _ = self.get_news(limit=1)
            return True
        except Exception:
            return False


def get_news_provider() -> NewsProvider:
    """
    Factory function to get the configured news provider.
    
    Returns:
        NewsProvider instance
    """
    config = get_config()
    provider_name = config.news_provider
    api_key = config.get('data_sources.news.api_key')
    
    if provider_name == 'mock':
        return MockNewsProvider()
    elif provider_name == 'finnhub':
        if not api_key or api_key.startswith('${'):
            print("Warning: Finnhub API key not set, falling back to mock provider")
            return MockNewsProvider()
        return FinnhubNewsProvider(
            api_key=api_key,
            base_url=config.get('data_sources.news.base_url', 'https://finnhub.io/api/v1')
        )
    elif provider_name == 'newsapi':
        if not api_key or api_key.startswith('${'):
            print("Warning: NewsAPI key not set, falling back to mock provider")
            return MockNewsProvider()
        return NewsAPIProvider(
            api_key=api_key,
            base_url=config.get('data_sources.news.base_url', 'https://newsapi.org/v2')
        )
    else:
        raise ValueError(f"Unsupported news provider: {provider_name}")

