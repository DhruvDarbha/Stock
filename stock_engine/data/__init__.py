"""Data access layer for market data, news, and user data."""

from .market_data import MarketDataProvider, get_market_data_provider
from .news_data import NewsProvider, get_news_provider
from .user_data import UserDataStore, get_user_data_store

__all__ = [
    'MarketDataProvider',
    'get_market_data_provider',
    'NewsProvider',
    'get_news_provider',
    'UserDataStore',
    'get_user_data_store',
]

