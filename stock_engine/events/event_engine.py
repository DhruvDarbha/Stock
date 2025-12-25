"""Main event processing engine."""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .event_classifier import EventClassifier
from ..data import get_news_provider
from ..config import get_config


class EventEngine:
    """Main engine for processing and tracking events."""
    
    def __init__(self):
        """Initialize event engine."""
        self.classifier = EventClassifier()
        self.news_provider = get_news_provider()
        self.config = get_config()
        
        # Event state cache (in production, would be in database)
        self._event_state: Dict[str, Dict] = {}
    
    def process_ticker_events(
        self,
        ticker: str,
        as_of_date: date,
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Process events for a ticker and compute event-based features.
        
        Args:
            ticker: Stock ticker
            as_of_date: Date to compute features as of
            lookback_days: Days to look back for events
        
        Returns:
            Dictionary with event features
        """
        start_date = as_of_date - timedelta(days=lookback_days)
        
        # Get and classify news
        news_df = self.news_provider.get_news(
            ticker=ticker,
            start_date=start_date,
            end_date=as_of_date,
            limit=500
        )
        
        if news_df.empty:
            return self._empty_event_features()
        
        # Classify events
        events = []
        for _, row in news_df.iterrows():
            event = self.classifier.classify_article(row.to_dict())
            events.append(event)
        
        events_df = pd.DataFrame(events)
        
        # Compute event features
        features = self._compute_event_features(events_df, as_of_date)
        
        # Update event state
        self._update_event_state(ticker, events_df, as_of_date)
        
        return features
    
    def _empty_event_features(self) -> Dict[str, Any]:
        """Return empty event features."""
        return {
            'event_intensity_7d': 0.0,
            'event_intensity_30d': 0.0,
            'recent_negative_events': 0,
            'recent_positive_events': 0,
            'high_impact_event_active': False,
            'latest_event_type': 'none',
            'latest_event_sentiment': 'neutral',
            'time_since_latest_event_days': 999,
            'shock_indicator': 0.0
        }
    
    def _compute_event_features(
        self,
        events_df: pd.DataFrame,
        as_of_date: date
    ) -> Dict[str, Any]:
        """Compute aggregated event features."""
        if events_df.empty:
            return self._empty_event_features()
        
        # Convert timestamps
        events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date
        
        # Filter by time windows
        date_7d = as_of_date - timedelta(days=7)
        date_30d = as_of_date - timedelta(days=30)
        
        events_7d = events_df[events_df['date'] >= date_7d]
        events_30d = events_df[events_df['date'] >= date_30d]
        
        # Event intensity (weighted by impact score)
        intensity_7d = events_7d['impact_score'].mean() if not events_7d.empty else 0.0
        intensity_30d = events_30d['impact_score'].mean() if not events_30d.empty else 0.0
        
        # Sentiment-based event counts
        sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        events_7d = events_7d.copy()  # Fix SettingWithCopyWarning
        events_30d = events_30d.copy()  # Fix SettingWithCopyWarning
        events_7d['sentiment_numeric'] = events_7d['sentiment'].map(sentiment_map)
        events_30d['sentiment_numeric'] = events_30d['sentiment'].map(sentiment_map)
        
        recent_negative = (events_7d['sentiment_numeric'] < 0).sum()
        recent_positive = (events_7d['sentiment_numeric'] > 0).sum()
        
        # High-impact event indicator
        high_impact_threshold = self.config.get('events.high_impact_sentiment_threshold', -0.7)
        high_impact_events = events_7d[
            (events_7d['impact_score'] >= 8.0) & 
            (events_7d['sentiment_numeric'] <= high_impact_threshold)
        ]
        high_impact_active = len(high_impact_events) > 0
        
        # Latest event
        latest_event = events_df.iloc[-1] if not events_df.empty else None
        latest_type = latest_event['event_type'] if latest_event is not None else 'none'
        latest_sentiment = latest_event['sentiment'] if latest_event is not None else 'neutral'
        
        if latest_event is not None:
            time_since = (as_of_date - latest_event['date']).days
        else:
            time_since = 999
        
        # Shock indicator: spike in event volume
        if len(events_df) >= 30:
            recent_volume = len(events_7d)
            historical_volume = len(events_df[events_df['date'] < date_7d]) / 23  # Average per day
            if historical_volume > 0:
                shock_zscore = (recent_volume / 7 - historical_volume) / (historical_volume ** 0.5 + 1)
            else:
                shock_zscore = recent_volume / 7
        else:
            shock_zscore = len(events_7d) / 7
        
        return {
            'event_intensity_7d': intensity_7d,
            'event_intensity_30d': intensity_30d,
            'recent_negative_events': recent_negative,
            'recent_positive_events': recent_positive,
            'high_impact_event_active': int(high_impact_active),
            'latest_event_type': latest_type,
            'latest_event_sentiment': latest_sentiment,
            'time_since_latest_event_days': time_since,
            'shock_indicator': shock_zscore
        }
    
    def _update_event_state(
        self,
        ticker: str,
        events_df: pd.DataFrame,
        as_of_date: date
    ):
        """Update internal event state cache."""
        if events_df.empty:
            return
        
        latest = events_df.iloc[-1]
        
        self._event_state[ticker] = {
            'latest_event': latest.to_dict(),
            'last_updated': as_of_date,
            'total_events_90d': len(events_df)
        }
    
    def get_event_state(self, ticker: str) -> Optional[Dict]:
        """Get current event state for a ticker."""
        return self._event_state.get(ticker)
    
    def should_apply_risk_multiplier(
        self,
        ticker: str,
        as_of_date: date
    ) -> Tuple[bool, float]:
        """
        Determine if risk multiplier should be applied due to extreme events.
        
        Returns:
            Tuple of (should_apply, multiplier)
        """
        features = self.process_ticker_events(ticker, as_of_date)
        
        if features['high_impact_event_active']:
            # Apply risk multiplier for high-impact negative events
            multiplier = 1.5 + (features['event_intensity_7d'] / 10.0)
            return True, multiplier
        
        # Check shock indicator
        shock_threshold = self.config.get('events.shock_threshold_zscore', 2.0)
        if features['shock_indicator'] > shock_threshold and features['recent_negative_events'] > 0:
            multiplier = 1.2 + (features['shock_indicator'] / 10.0)
            return True, multiplier
        
        return False, 1.0
    
    def process_multiple_tickers(
        self,
        tickers: List[str],
        as_of_date: date,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Process events for multiple tickers.
        
        Returns:
            DataFrame with ticker as index and event features as columns
        """
        results = []
        
        for ticker in tickers:
            try:
                features = self.process_ticker_events(ticker, as_of_date, lookback_days)
                features['ticker'] = ticker
                results.append(features)
            except Exception as e:
                print(f"Error processing events for {ticker}: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df.set_index('ticker', inplace=True)
        return df

