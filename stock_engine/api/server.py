"""FastAPI server for the stock recommendation engine."""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
from pathlib import Path

from ..models import MarketModel, BehaviorModel
from ..portfolio import PortfolioOptimizer, TradeGenerator
from ..events import EventEngine
from ..config import get_config
from ..data import get_user_data_store


app = FastAPI(title="Stock Recommendation Engine API", version="0.1.0")


# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: int
    portfolio_value: float
    tickers: Optional[List[str]] = None
    as_of_date: Optional[date] = None


class RecommendationResponse(BaseModel):
    ticker: str
    action: str
    shares: int
    current_weight: float
    target_weight: float
    expected_excess_return: float
    risk_score: float
    rationale: str


class Trade(BaseModel):
    ticker: str
    action: str
    shares: int
    current_weight: float
    target_weight: float
    rationale: str


# Global models (would be loaded on startup in production)
_market_model: Optional[MarketModel] = None
_behavior_model: Optional[BehaviorModel] = None
_event_engine: Optional[EventEngine] = None
_optimizer: Optional[PortfolioOptimizer] = None
_trade_generator: Optional[TradeGenerator] = None


def load_models():
    """Load models (called on startup)."""
    global _market_model, _behavior_model, _event_engine, _optimizer, _trade_generator
    
    config = get_config()
    model_type = config.get('models.market_model.type', 'lightgbm')
    model_dir = Path('models/saved')
    
    # Load market model
    market_model_path = None
    for path in model_dir.glob(f'market_model_{model_type}_*.pkl'):
        market_model_path = path
        break
    
    if market_model_path and market_model_path.exists():
        _market_model = MarketModel(model_type=model_type)
        _market_model.load(str(market_model_path))
    
    # Load behavior model (optional)
    behavior_model_path = model_dir / f'behavior_model_{model_type}_follow.pkl'
    if behavior_model_path.exists():
        _behavior_model = BehaviorModel(model_type=model_type)
        _behavior_model.load(str(behavior_model_path))
    
    # Initialize event engine
    _event_engine = EventEngine()
    
    # Initialize optimizer and trade generator
    if _market_model:
        _optimizer = PortfolioOptimizer(
            market_model=_market_model,
            behavior_model=_behavior_model,
            event_engine=_event_engine
        )
        _trade_generator = TradeGenerator(_optimizer)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


def get_trade_generator() -> TradeGenerator:
    """Dependency to get trade generator."""
    if _trade_generator is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    return _trade_generator


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Recommendation Engine API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "market_model_loaded": _market_model is not None and _market_model.is_trained,
        "behavior_model_loaded": _behavior_model is not None and _behavior_model.is_trained
    }


@app.post("/recommendations", response_model=List[Trade])
async def get_recommendations(
    request: RecommendationRequest,
    generator: TradeGenerator = Depends(get_trade_generator)
):
    """
    Get stock recommendations for a user.
    
    Args:
        request: Recommendation request with user_id, portfolio_value, etc.
        generator: Trade generator dependency
    
    Returns:
        List of trade recommendations
    """
    user_store = get_user_data_store()
    user = user_store.get_user(request.user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
    
    # Default tickers if not provided
    tickers = request.tickers or [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD'
    ]
    
    as_of_date = request.as_of_date or date.today()
    
    # Generate trades
    trades_df = generator.generate_trades(
        user_id=request.user_id,
        tickers=tickers,
        portfolio_value=request.portfolio_value,
        as_of_date=as_of_date
    )
    
    if trades_df.empty:
        return []
    
    # Convert to response format
    trades = []
    for _, row in trades_df.iterrows():
        if row['action'] == 'hold':
            continue
        
        trades.append(Trade(
            ticker=row['ticker'],
            action=row['action'],
            shares=int(row['shares']),
            current_weight=float(row['current_weight']),
            target_weight=float(row['target_weight']),
            rationale=row.get('rationale', '')
        ))
    
    return trades


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user information."""
    user_store = get_user_data_store()
    user = user_store.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    return user


@app.get("/users/{user_id}/positions")
async def get_positions(user_id: int):
    """Get user positions."""
    user_store = get_user_data_store()
    positions = user_store.get_positions(user_id)
    
    return positions.to_dict('records')


@app.get("/users/{user_id}/recommendations")
async def get_user_recommendations(user_id: int, limit: int = 10):
    """Get user's recommendation history."""
    user_store = get_user_data_store()
    recommendations = user_store.get_recommendations(user_id, limit=limit)
    
    return recommendations.to_dict('records')


def create_app() -> FastAPI:
    """Create and return FastAPI app."""
    return app

