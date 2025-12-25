# Stock Recommendation Engine - Architecture Documentation

## Overview

This document describes the architecture of the Stock Recommendation Engine, a production-grade system for predicting risk-adjusted stock returns, modeling user behavior, and generating personalized portfolio recommendations.

## System Goals

The system produces daily recommendations like:
- "Buy 25 shares of AAPL"
- "Sell 10 shares of KO"
- "Hold MSFT"

Based on:
1. Predicted 6-12 month excess returns vs benchmark
2. Predicted 1-3 month drawdown risk
3. User risk tolerance, investment horizon, and constraints
4. Historical user behavior (follow-through, panic behavior)
5. Current events and news triggers

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Market Data  │  │  News Data   │  │  User Data   │         │
│  │  Providers   │  │  Providers   │  │    Store     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Engineering                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Stock      │  │     User     │  │    Event     │         │
│  │  Features    │  │   Features   │  │   Features   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Model Layer                               │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │    Market    │  │   Behavior   │                            │
│  │    Model     │  │    Model     │                            │
│  │ (Returns)    │  │ (Follow/Exit)│                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Portfolio Optimization                         │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │  Optimizer   │  │    Trade     │                            │
│  │ (Weights)    │  │  Generator   │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│                     (REST API / CLI)                             │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

#### Market Data Providers (`stock_engine/data/market_data.py`)

**Purpose**: Abstract interface for fetching stock price and fundamental data.

**Supported Providers**:
- **Mock**: For testing/development (generates synthetic data)
- **Polygon.io**: Production market data API
- **TODO**: Alpha Vantage, Finnhub, Tiingo

**Key Methods**:
- `get_ohlcv(ticker, start_date, end_date)` → OHLCV DataFrame
- `get_fundamentals(ticker)` → Dict with P/E, revenue, margins, etc.
- `get_multiple_ohlcv(tickers, ...)` → Batch fetching

**Configuration**: Set provider in `config/config.yaml`:
```yaml
data_sources:
  market_data:
    provider: "polygon"  # or "mock"
    api_key: "${MARKET_DATA_API_KEY}"
```

#### News Providers (`stock_engine/data/news_data.py`)

**Purpose**: Fetch financial news and sentiment data.

**Supported Providers**:
- **Mock**: Synthetic news for testing
- **Finnhub**: Production news API
- **TODO**: NewsAPI, FinancialModelingPrep

**Key Methods**:
- `get_news(ticker, start_date, end_date)` → News articles DataFrame
- `get_news_mentions(ticker, ...)` → Time series of mention counts/sentiment

#### User Data Store (`stock_engine/data/user_data.py`)

**Purpose**: Database abstraction for user and portfolio data.

**Schema**:
- `users`: User profiles (risk tolerance, horizon, constraints)
- `positions`: Current stock positions
- `recommendations`: Generated recommendations
- `user_actions`: What users actually did

**Database**: SQLite (default), can be switched to PostgreSQL/MySQL

### 2. Feature Engineering

#### Stock Features (`stock_engine/features/stock_features.py`)

Computes features for each stock-date:

**Technical Features**:
- Returns: 1d, 5d, 1m, 3m, 12m
- Moving averages: 20/50/200-day, distance from MAs
- Volatility: 20d/60d annualized, ATR
- Price levels: distance from 52-week high/low
- Volume: average volume, volume ratio

**Fundamental Features**:
- Valuation: P/E, EV/EBITDA, P/S (vs history, vs sector)
- Growth: Revenue/Earnings growth (YoY, QoQ)
- Profitability: Margins, ROE, ROIC
- Leverage: Debt-to-equity

**Sentiment Features**:
- News sentiment: 7d/30d average
- Mention counts: 7d/30d
- Shock indicators: Z-score of mention volume

**Event Features**:
- Recent negative/positive events
- High-impact event flags
- Time since latest event

#### User Features (`stock_engine/features/user_features.py`)

Computes features for each user:

**Static Features**:
- Risk profile: Conservative/Moderate/Aggressive (one-hot)
- Investment horizon: Short/Medium/Long (one-hot)
- Constraints: Max single-stock weight, sector caps

**Behavioral Features**:
- Follow rate: % of recommendations followed
- Average holding period
- Max drawdown historically tolerated
- Volatility sensitivity
- Preferred turnover

### 3. Event Processing (`stock_engine/events/`)

#### Event Classifier (`event_classifier.py`)

**Purpose**: Extract structured events from news articles.

**Event Types**:
- Endorsement (celebrity/influencer)
- Controversy/Boycott
- Product issues/Recalls
- Lawsuits
- Executive changes
- Regulatory actions
- Earnings

**Features**:
- Rule-based classification (patterns/keywords)
- Sentiment extraction (keyword-based, TODO: ML/NLP)
- Entity extraction (people, companies, products)
- Impact scoring (1-10 scale)

**TODO**: Integrate LLM (OpenAI, Anthropic) for better extraction

#### Event Engine (`event_engine.py`)

**Purpose**: Process and track events, compute event-based risk adjustments.

**Key Functions**:
- `process_ticker_events(ticker, as_of_date)` → Event features
- `should_apply_risk_multiplier(ticker, date)` → (bool, multiplier)
- Maintains event state cache (latest event, intensity, time since)

**Risk Multiplier Logic**:
- High-impact negative events → Apply 1.5x+ risk multiplier
- Shock indicators (spike in mentions) → Apply 1.2x+ multiplier
- Decays over time (90 days)

### 4. Models

#### Market Model (`stock_engine/models/market_model.py`)

**Purpose**: Predict excess returns and risk for stocks.

**Architecture**:
- Gradient boosting (LightGBM/XGBoost/CatBoost)
- Input: Stock features (technical, fundamental, sentiment, events)
- Output: Predicted 6-12 month excess return vs benchmark
- Also estimates: Downside risk (max drawdown proxy)

**Training**:
- Target: Future excess return = (Stock return - Benchmark return) over horizon
- Features computed as of each date, no look-ahead bias
- Train on historical data (e.g., 2010-2023)

**Prediction**:
- `predict(X)` → Expected excess returns
- `predict_risk(X)` → Risk scores (simplified heuristic, TODO: separate risk model)

#### Behavior Model (`stock_engine/models/behavior_model.py`)

**Purpose**: Predict user behavior (follow probability, panic exits).

**Tasks**:
1. **Follow Prediction**: P(user follows recommendation)
   - Input: User features + Recommendation features (ticker, size, direction)
   - Output: Probability (0-1)
   - Target: Binary (1 if followed, 0 if not)

2. **Panic Exit Prediction**: P(user sells under stress)
   - Input: User features + Current drawdown/volatility
   - Output: Probability of panic exit
   - Target: Binary (1 if early exit, 0 if held)

**Architecture**:
- Gradient boosting classifier (LightGBM/XGBoost/CatBoost)
- Baseline: Logistic regression

**TODO**: Sequence models (RNN/Transformer) for temporal behavior patterns

### 5. Portfolio Optimization (`stock_engine/portfolio/`)

#### Optimizer (`optimizer.py`)

**Purpose**: Compute optimal portfolio weights.

**Algorithm** (simplified):
1. Get predictions: Expected excess returns, risk scores
2. Apply event-based risk multipliers if needed
3. Compute risk-adjusted scores: `score = expected_return / risk`
4. Adjust for user behavior: `effective_score = score * P(follow)`
5. Allocate weights:
   - Rank by score
   - Allocate proportional to score, subject to constraints
   - Max single-stock weight (10%)
   - Sector caps (30%)
   - Min trade size (0.5%)

**TODO**: Replace with proper optimization:
- Mean-variance optimization (Markowitz)
- Risk parity
- Black-Litterman
- Constrained optimization (CVXPY)

#### Trade Generator (`rebalancer.py`)

**Purpose**: Convert target weights to concrete trades.

**Logic**:
1. Get current positions → current weights
2. Compare with target weights
3. Generate trades if difference > rebalance threshold (1%)
4. Respect constraints:
   - Max turnover per rebalance (20%)
   - Min trade size (0.5% of portfolio)
   - Round shares to integers
5. Generate rationale for each trade

**Output**: DataFrame with:
- Ticker, action (buy/sell/hold), shares
- Current/target weights
- Expected return, risk score
- Rationale

### 6. Data Pipeline

#### Training Pipeline

1. **Market Model Training** (`scripts/train_market_model.py`):
   - Load historical data (prices, fundamentals, news)
   - Compute features for each stock-date
   - Compute targets (future excess returns)
   - Train model, save to `models/saved/`

2. **Behavior Model Training** (`scripts/train_behavior_model.py`):
   - Load user recommendation/action history
   - Compute user + recommendation features
   - Train on follow/panic targets
   - Save to `models/saved/`

#### Inference Pipeline

1. **Daily Recommendation Generation** (`scripts/generate_recommendations.py`):
   - Load trained models
   - For each user:
     - Get current positions
     - Compute features for ticker universe
     - Get predictions from market model
     - Adjust for user behavior
     - Optimize weights
     - Generate trades
     - Save recommendations to DB

2. **Event Processing**:
   - Continuously ingest news (scheduled job)
   - Classify events
   - Update event state
   - Trigger risk adjustments

### 7. Backtesting (`stock_engine/backtesting/`)

#### Backtester (`backtester.py`)

**Purpose**: Evaluate strategy performance on historical data.

**Process**:
1. Initialize portfolio with starting capital
2. For each rebalance date:
   - Compute features as of that date
   - Get model predictions (no look-ahead)
   - Optimize weights
   - Execute trades
   - Track portfolio value
3. Compare vs benchmark (S&P 500)

**Metrics**:
- CAGR (Compound Annual Growth Rate)
- Sharpe ratio
- Max drawdown
- Volatility
- Excess return vs benchmark

#### Evaluator (`evaluator.py`)

**Purpose**: Compute model evaluation metrics.

**Market Model Metrics**:
- Correlation (Pearson, Spearman)
- MSE, MAE, RMSE
- Hit rate (top decile predictions)
- Calibration (for risk predictions)

**Behavior Model Metrics**:
- Accuracy, Precision, Recall, F1
- AUC-ROC (TODO)

### 8. API Layer (`stock_engine/api/`)

**Framework**: FastAPI

**Endpoints**:
- `GET /` → API info
- `GET /health` → Health check
- `POST /recommendations` → Get recommendations for user
- `GET /users/{user_id}` → Get user info
- `GET /users/{user_id}/positions` → Get positions
- `GET /users/{user_id}/recommendations` → Get recommendation history

**Run**: `uvicorn stock_engine.api.server:app --host 0.0.0.0 --port 8000`

## Configuration

All configuration in `config/config.yaml`:

```yaml
# Data sources
data_sources:
  market_data:
    provider: "polygon"  # or "mock"
    api_key: "${MARKET_DATA_API_KEY}"
  news:
    provider: "finnhub"
    api_key: "${NEWS_API_KEY}"

# Models
models:
  market_model:
    type: "lightgbm"
    target_horizon_days: 252  # 12 months

# Portfolio
portfolio:
  benchmark_ticker: "SPY"
  max_single_stock_weight: 0.10
  rebalance_threshold: 0.01
```

Set API keys in `.env` file (see `.env.example`).

## File Structure

```
Stock/
├── config/
│   └── config.yaml           # Configuration
├── scripts/
│   ├── train_market_model.py
│   ├── train_behavior_model.py
│   └── generate_recommendations.py
├── stock_engine/
│   ├── __init__.py
│   ├── config.py             # Config loader
│   ├── data/                 # Data access layer
│   ├── features/             # Feature engineering
│   ├── events/               # Event processing
│   ├── models/               # ML models
│   ├── portfolio/            # Optimization
│   ├── backtesting/          # Backtesting
│   └── api/                  # API server
├── models/
│   └── saved/                # Saved models
├── data/
│   └── stock_engine.db       # SQLite database
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

## Production Considerations

### TODO: Production Improvements

1. **Data Providers**:
   - Implement real API integrations (Polygon, Finnhub)
   - Add caching layer (Redis)
   - Add retry logic, rate limiting

2. **Models**:
   - Implement proper risk model (quantile regression)
   - Add sequence models for temporal patterns
   - Model versioning and A/B testing

3. **Optimization**:
   - Replace simple allocation with proper optimization (CVXPY)
   - Add transaction cost models
   - Tax-loss harvesting

4. **Events**:
   - Integrate LLM for better event extraction
   - Real-time news ingestion (streaming)
   - Social media integration (Twitter, Reddit)

5. **Scaling**:
   - Move to PostgreSQL/MySQL for production
   - Add caching (Redis)
   - Distributed training (Dask, Ray)
   - API rate limiting, authentication

6. **Monitoring**:
   - Model performance tracking
   - Recommendation quality metrics
   - User engagement metrics
   - Error logging and alerting

7. **Security**:
   - API authentication (JWT, OAuth)
   - Encrypt sensitive data
   - Secure API key storage (secrets manager)

## Usage Examples

### Train Models

```bash
# Train market model
python scripts/train_market_model.py

# Train behavior model (requires user history)
python scripts/train_behavior_model.py
```

### Generate Recommendations

```bash
python scripts/generate_recommendations.py --user-id 1 --portfolio-value 100000
```

### Run API

```bash
uvicorn stock_engine.api.server:app --host 0.0.0.0 --port 8000
```

### Backtest

```python
from stock_engine.models import MarketModel
from stock_engine.backtesting import Backtester
from datetime import date

# Load model
model = MarketModel()
model.load('models/saved/market_model_lightgbm_2023-12-31.pkl')

# Run backtest
backtester = Backtester(model)
results = backtester.run_backtest(
    tickers=['AAPL', 'GOOGL', 'MSFT'],
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31)
)

# Compute metrics
metrics = backtester.compute_metrics(results)
print(metrics)
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- pandas, numpy: Data processing
- lightgbm, xgboost, catboost: Models
- fastapi, uvicorn: API
- sqlalchemy: Database
- requests: HTTP clients

## License

[Add license information]

