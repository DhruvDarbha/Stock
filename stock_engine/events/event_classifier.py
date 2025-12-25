"""Event classification and extraction from news."""

import pandas as pd
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import re

from ..data import get_news_provider
from ..config import get_config


class EventClassifier:
    """Classify and extract structured events from news."""
    
    # Event type patterns (can be enhanced with ML/LLM)
    EVENT_PATTERNS = {
        'endorsement': [
            r'endors',
            r'partner',
            r'sponsor',
            r'ambassador',
            r'celebrity',
            r'influencer'
        ],
        'controversy': [
            r'controvers',
            r'scandal',
            r'boycott',
            r'protest',
            r'outrage'
        ],
        'product_issue': [
            r'recall',
            r'defect',
            r'issue',
            r'problem',
            r'safety',
            r'fault'
        ],
        'lawsuit': [
            r'lawsuit',
            r'legal action',
            r'sue',
            r'litigation',
            r'court',
            r'judge'
        ],
        'executive_change': [
            r'CEO',
            r'CFO',
            r'executive',
            r'leadership',
            r'resign',
            r'appoint'
        ],
        'regulatory': [
            r'regulator',
            r'FDA',
            r'SEC',
            r'investigat',
            r'approval',
            r'ban'
        ],
        'earnings': [
            r'earnings',
            r'profit',
            r'revenue',
            r'quarter',
            r'guidance'
        ]
    }
    
    def __init__(self):
        """Initialize event classifier."""
        self.news_provider = get_news_provider()
        self.config = get_config()
    
    def classify_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a news article and extract event information.
        
        Args:
            article: Dictionary with keys: headline, content, timestamp, tickers
        
        Returns:
            Dictionary with: event_type, sentiment, entities, impact_score
        """
        text = f"{article.get('headline', '')} {article.get('content', '')}".lower()
        
        # Detect event type
        event_type = self._detect_event_type(text)
        
        # Extract sentiment (can be enhanced with sentiment analysis model)
        sentiment = self._extract_sentiment(article)
        
        # Extract entities (simplified - can use NER model)
        entities = self._extract_entities(text, article.get('tickers', []))
        
        # Estimate impact score (1-10 scale)
        impact_score = self._estimate_impact(event_type, sentiment, text)
        
        return {
            'event_type': event_type,
            'sentiment': sentiment,
            'entities': entities,
            'impact_score': impact_score,
            'timestamp': article.get('timestamp'),
            'tickers': article.get('tickers', [])
        }
    
    def _detect_event_type(self, text: str) -> str:
        """Detect event type from text."""
        scores = {}
        
        for event_type, patterns in self.EVENT_PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text, re.IGNORECASE))
            if score > 0:
                scores[event_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'general'
    
    def _extract_sentiment(self, article: Dict[str, Any]) -> str:
        """
        Extract sentiment from article.
        
        TODO: Replace with proper sentiment analysis model (e.g., VADER, transformer-based)
        """
        # Use existing sentiment if available
        if 'sentiment' in article:
            return article['sentiment']
        
        # Simple keyword-based sentiment (placeholder)
        text = f"{article.get('headline', '')} {article.get('content', '')}".lower()
        
        positive_words = ['good', 'great', 'excellent', 'strong', 'growth', 'profit', 'up', 'gain', 'beat']
        negative_words = ['bad', 'poor', 'weak', 'loss', 'down', 'drop', 'miss', 'decline', 'fall']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_entities(self, text: str, tickers: List[str]) -> Dict[str, List[str]]:
        """
        Extract entities from text.
        
        TODO: Use proper NER model (e.g., spaCy, transformers)
        """
        entities = {
            'people': [],
            'companies': tickers.copy() if tickers else [],
            'products': []
        }
        
        # Placeholder - would use NER model in production
        # Common celebrity/influencer names (very simplified)
        celebrity_patterns = [
            r'\b(ronaldo|cristiano|messi|lebron|curry|taylor swift|beyonce|elon musk|bezos|gates)\b'
        ]
        
        for pattern in celebrity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['people'].extend(matches)
        
        return entities
    
    def _estimate_impact(
        self,
        event_type: str,
        sentiment: str,
        text: str
    ) -> float:
        """
        Estimate event impact score (1-10).
        
        Higher scores indicate more significant impact.
        """
        base_scores = {
            'endorsement': 6.0,
            'controversy': 8.0,
            'product_issue': 7.0,
            'lawsuit': 8.5,
            'executive_change': 7.5,
            'regulatory': 8.0,
            'earnings': 5.0,
            'general': 3.0
        }
        
        score = base_scores.get(event_type, 3.0)
        
        # Adjust for sentiment
        if sentiment == 'negative':
            score *= 1.2  # Negative events often have more impact
        elif sentiment == 'positive':
            score *= 1.1
        
        # Adjust for intensity keywords
        intensity_keywords = ['major', 'significant', 'huge', 'massive', 'critical', 'urgent']
        if any(keyword in text for keyword in intensity_keywords):
            score *= 1.3
        
        return min(score, 10.0)
    
    def process_news_batch(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Process a batch of news articles and classify events.
        
        Returns:
            DataFrame with classified events
        """
        news_df = self.news_provider.get_news(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        if news_df.empty:
            return pd.DataFrame()
        
        events = []
        for _, row in news_df.iterrows():
            article = row.to_dict()
            event = self.classify_article(article)
            events.append(event)
        
        return pd.DataFrame(events)


class LLMEventClassifier(EventClassifier):
    """
    Enhanced event classifier using LLM for better extraction.
    
    TODO: Implement with OpenAI API or local LLM for production use.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM-based classifier."""
        super().__init__()
        self.api_key = api_key
        # TODO: Initialize LLM client (OpenAI, Anthropic, etc.)
    
    def classify_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to classify article with structured extraction.
        
        TODO: Implement API call to LLM for structured event extraction.
        """
        # Placeholder - would call LLM API
        # Prompt: "Extract event type, sentiment, entities (people, companies), and impact from this article..."
        
        # For now, fall back to rule-based
        return super().classify_article(article)

