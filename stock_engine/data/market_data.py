"""Market data provider implementations."""

import pandas as pd
import requests
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from abc import ABC
import time

from .base import MarketDataProvider
from ..config import get_config


class MockMarketDataProvider(MarketDataProvider):
    """Mock market data provider for testing and development."""
    
    def __init__(self):
        """Initialize mock provider with sample data."""
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "day"
    ) -> pd.DataFrame:
        """Generate mock OHLCV data."""
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Generate synthetic price data
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = [d.date() for d in dates if d.weekday() < 5]  # Weekdays only
        
        # Simple random walk starting from $100
        import numpy as np
        np.random.seed(hash(ticker) % 2**32)
        
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * (1 + returns).cumprod()
        
        # Generate OHLC from close prices
        data = []
        for i, (d, close) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.5, 1.5)
            high = close * (1 + volatility * 0.01)
            low = close * (1 - volatility * 0.01)
            open_price = close * (1 + np.random.uniform(-0.005, 0.005))
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'date': d,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        self._cache[cache_key] = df
        return df
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Generate mock fundamental data."""
        import numpy as np
        np.random.seed(hash(ticker) % 2**32)
        
        return {
            'ticker': ticker,
            'market_cap': np.random.uniform(1e9, 1e12),
            'pe_ratio': np.random.uniform(10, 30),
            'ev_ebitda': np.random.uniform(5, 20),
            'ps_ratio': np.random.uniform(1, 10),
            'revenue_growth_yoy': np.random.uniform(-0.2, 0.3),
            'earnings_growth_yoy': np.random.uniform(-0.3, 0.4),
            'gross_margin': np.random.uniform(0.2, 0.6),
            'net_margin': np.random.uniform(0.05, 0.25),
            'roe': np.random.uniform(0.05, 0.25),
            'roic': np.random.uniform(0.05, 0.30),
            'debt_to_equity': np.random.uniform(0, 2),
            'last_earnings_date': (date.today() - timedelta(days=30)).isoformat(),
            'next_earnings_date': (date.today() + timedelta(days=60)).isoformat(),
        }
    
    def get_multiple_ohlcv(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        interval: str = "day"
    ) -> Dict[str, pd.DataFrame]:
        """Get OHLCV for multiple tickers."""
        result = {}
        for ticker in tickers:
            result[ticker] = self.get_ohlcv(ticker, start_date, end_date, interval)
        return result


class PolygonMarketDataProvider(MarketDataProvider):
    """Polygon.io market data provider."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io"):
        """
        Initialize Polygon provider.
        
        Args:
            api_key: Polygon.io API key
            base_url: Base URL for API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit_delay = 1.0 / get_config().get('data_sources.market_data.rate_limit', 5)
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _make_request_with_retry(self, url: str, params: dict, max_retries: int = 3):
        """Make API request with retry logic and better error handling."""
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, params=params, timeout=30)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 60  # Exponential backoff: 1min, 2min, 4min
                        print(f"Rate limited. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        response.raise_for_status()
                
                # Handle forbidden (403) - might be free tier limitation
                if response.status_code == 403:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', error_data.get('message', 'Forbidden'))
                    except:
                        error_msg = 'Forbidden - Your Polygon plan may not support this data'
                    raise ValueError(f"API access denied (403): {error_msg}. Your Polygon plan may not support this data.")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff
                    print(f"Request failed: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        
        raise Exception("Max retries exceeded")
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "day"
    ) -> pd.DataFrame:
        """Get OHLCV data from Polygon.io."""
        # Map interval to Polygon format
        timespan_map = {
            'day': 'day',
            'hour': 'hour',
            'minute': 'minute'
        }
        timespan = timespan_map.get(interval, 'day')
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date}/{end_date}"
        params = {'apiKey': self.api_key, 'adjusted': 'true', 'sort': 'asc'}
        
        response = self._make_request_with_retry(url, params)
        data = response.json()
        
        if data.get('status') != 'OK' or 'results' not in data:
            error_msg = data.get('error', data.get('status', 'Unknown'))
            raise ValueError(f"Polygon API error: {error_msg}")
        
        results = data['results']
        if not results:
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
        
        return pd.DataFrame({
            'date': df['date'],
            'open': df['o'],
            'high': df['h'],
            'low': df['l'],
            'close': df['c'],
            'volume': df['v']
        })
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get fundamentals from Polygon.io."""
        self._rate_limit()
        
        # TODO: Implement Polygon fundamentals endpoint
        # For now, return basic structure
        # Polygon has different endpoints for fundamentals - need to implement based on their API docs
        raise NotImplementedError("Polygon fundamentals endpoint - TODO: implement based on API docs")
    
    def get_multiple_ohlcv(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        interval: str = "day"
    ) -> Dict[str, pd.DataFrame]:
        """Get OHLCV for multiple tickers."""
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.get_ohlcv(ticker, start_date, end_date, interval)
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue
        return result
    
    def health_check(self) -> bool:
        """Check if Polygon API is accessible."""
        try:
            end = date.today()
            start = date(end.year, end.month, 1)
            _ = self.get_ohlcv("SPY", start, end)
            return True
        except Exception:
            return False


class AlphaVantageMarketDataProvider(MarketDataProvider):
    """Alpha Vantage market data provider (free tier available)."""
    
    def __init__(self, api_key: str, base_url: str = "https://www.alphavantage.co/query"):
        """
        Initialize Alpha Vantage provider.
        
        Args:
            api_key: Alpha Vantage API key
            base_url: Base URL for API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit_delay = 12.0  # Free tier: 5 requests/minute = 1 per 12 seconds
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting (5 requests/minute for free tier)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def get_ohlcv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        interval: str = "day"
    ) -> pd.DataFrame:
        """Get OHLCV data from Alpha Vantage."""
        self._rate_limit()
        
        # Alpha Vantage TIME_SERIES_DAILY endpoint (free tier)
        # Note: 'full' outputsize is premium, so we use 'compact' (last 100 data points)
        url = self.base_url
        params = {
            'function': 'TIME_SERIES_DAILY',  # Free tier endpoint
            'symbol': ticker,
            'apikey': self.api_key,
            'outputsize': 'compact',  # Free tier: last 100 data points only
            'datatype': 'json'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if 'Note' in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
        if 'Information' in data:
            # Check if it's about premium endpoint
            info = data['Information']
            if 'premium' in info.lower():
                raise ValueError(f"Alpha Vantage: {info} - This shouldn't happen with TIME_SERIES_DAILY")
            raise ValueError(f"Alpha Vantage: {info}")
        
        # Extract time series data
        time_series_key = 'Time Series (Daily)'
        if time_series_key not in data:
            raise ValueError(f"No time series data in response. Keys: {data.keys()}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        records = []
        for date_str, values in time_series.items():
            record_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # Filter by date range
            if record_date < start_date or record_date > end_date:
                continue
            
            records.append({
                'date': record_date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),  # No adjusted close in free tier
                'volume': int(values['5. volume'])
            })
        
        if not records:
            # Free tier limitation: only returns last 100 days
            days_ago = (date.today() - start_date).days
            if days_ago > 120:
                print(f"  ⚠️  Warning: Alpha Vantage free tier only returns last ~100 days. Requested start date {start_date} ({days_ago} days ago) is too old.")
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        df = pd.DataFrame(records)
        df = df.sort_values('date')
        df.reset_index(drop=True, inplace=True)
        
        # Warn if we got less data than expected (free tier limitation)
        if len(df) < 50:
            print(f"  ⚠️  Note: Alpha Vantage free tier returned only {len(df)} days (last ~100 days available)")
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get fundamentals from Alpha Vantage."""
        self._rate_limit()
        
        # Use OVERVIEW endpoint for fundamentals
        url = self.base_url
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        
        # Extract and convert fundamentals
        fundamentals = {
            'ticker': ticker,
            'market_cap': self._safe_float(data.get('MarketCapitalization')),
            'pe_ratio': self._safe_float(data.get('PERatio')),
            'ev_ebitda': self._safe_float(data.get('EVToEBITDA')),
            'ps_ratio': self._safe_float(data.get('PriceToSalesRatioTTM')),
            'revenue_growth_yoy': self._safe_float(data.get('QuarterlyRevenueGrowthYOY')),
            'earnings_growth_yoy': self._safe_float(data.get('QuarterlyEarningsGrowthYOY')),
            'gross_margin': self._safe_float(data.get('GrossProfitTTM')) / self._safe_float(data.get('RevenueTTM')) if self._safe_float(data.get('RevenueTTM')) else None,
            'net_margin': self._safe_float(data.get('ProfitMargin')),
            'roe': self._safe_float(data.get('ReturnOnEquityTTM')),
            'roic': None,  # Not directly available
            'debt_to_equity': self._safe_float(data.get('DebtToEquity')),
            'last_earnings_date': data.get('LatestQuarter'),
            'next_earnings_date': None  # Not available
        }
        
        return fundamentals
    
    def _safe_float(self, value):
        """Safely convert to float, handling None and 'None' string."""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def get_multiple_ohlcv(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        interval: str = "day"
    ) -> Dict[str, pd.DataFrame]:
        """Get OHLCV for multiple tickers."""
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.get_ohlcv(ticker, start_date, end_date, interval)
                # Additional delay between tickers for free tier
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue
        return result
    
    def health_check(self) -> bool:
        """Check if Alpha Vantage API is accessible."""
        try:
            # Quick test with a common ticker
            end = date.today()
            start = date(end.year, end.month, 1)
            _ = self.get_ohlcv("IBM", start, end)
            return True
        except Exception:
            return False


def get_market_data_provider() -> MarketDataProvider:
    """
    Factory function to get the configured market data provider.
    
    Returns:
        MarketDataProvider instance
    """
    config = get_config()
    provider_name = config.market_data_provider
    api_key = config.get('data_sources.market_data.api_key')
    
    if provider_name == 'mock':
        return MockMarketDataProvider()
    elif provider_name == 'polygon':
        if not api_key or api_key.startswith('${'):
            print("Warning: Polygon API key not set, falling back to mock provider")
            return MockMarketDataProvider()
        return PolygonMarketDataProvider(
            api_key=api_key,
            base_url=config.get('data_sources.market_data.base_url', 'https://api.polygon.io')
        )
    elif provider_name == 'alpha_vantage':
        if not api_key or api_key.startswith('${'):
            print("Warning: Alpha Vantage API key not set, falling back to mock provider")
            return MockMarketDataProvider()
        return AlphaVantageMarketDataProvider(
            api_key=api_key,
            base_url=config.get('data_sources.market_data.base_url', 'https://www.alphavantage.co/query')
        )
    else:
        raise ValueError(f"Unsupported market data provider: {provider_name}")

