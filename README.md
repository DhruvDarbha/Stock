# Stock Recommendation Engine

A production-grade system for predicting risk-adjusted stock returns, modeling user behavior, and generating personalized portfolio recommendations.

## Features

- **Market Prediction**: Predicts 6-12 month excess returns and risk for individual stocks using gradient boosting models
- **User Behavior Modeling**: Models user risk tolerance, follow-through, and panic behavior
- **Event Processing**: Incorporates current events and news triggers (endorsements, controversies, product issues, etc.)
- **Portfolio Optimization**: Generates concrete buy/sell/hold recommendations with position sizing
- **Backtesting**: Evaluate strategy performance on historical data
- **REST API**: FastAPI-based API for integration

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Stock
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional - works with mock providers):
```bash
# The .env file is already created
# Edit .env and add your API keys for real data
# See API_KEYS_GUIDE.md for detailed instructions
```

**Note**: The system works out of the box with **mock providers** (no API keys needed for testing). See [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md) for details on getting API keys for production use.

4. Configure the system:
```bash
# Edit config/config.yaml to set data providers, model types, etc.
```

### Train Models

```bash
# Train market prediction model
python scripts/train_market_model.py

# Train behavior model (requires user history)
python scripts/train_behavior_model.py
```

### Generate Recommendations

```bash
# Generate recommendations for a user
python scripts/generate_recommendations.py --user-id 1 --portfolio-value 100000
```

### Run API Server

```bash
uvicorn stock_engine.api.server:app --host 0.0.0.0 --port 8000
```

Then access the API at `http://localhost:8000/docs` for interactive documentation.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture and component documentation.

## Key Components

### Data Layer
- **Market Data Providers**: Fetch OHLCV and fundamental data (Polygon, Alpha Vantage, or mock)
- **News Providers**: Fetch financial news and sentiment (Finnhub, NewsAPI, or mock)
- **User Data Store**: SQLite/PostgreSQL database for users, positions, recommendations

### Feature Engineering
- **Stock Features**: Technical indicators, fundamentals, sentiment, events
- **User Features**: Risk profile, investment horizon, behavioral metrics

### Models
- **Market Model**: Predicts excess returns and risk (LightGBM/XGBoost/CatBoost)
- **Behavior Model**: Predicts user follow probability and panic exits

### Portfolio Optimization
- **Optimizer**: Computes optimal portfolio weights under constraints
- **Trade Generator**: Converts target weights to concrete buy/sell recommendations

### Event Processing
- **Event Classifier**: Extracts structured events from news (endorsements, controversies, etc.)
- **Event Engine**: Tracks events and applies risk adjustments

## Configuration

Main configuration file: `config/config.yaml`

Key settings:
- Data providers (defaults to "mock" for testing - works without API keys)
- Model types (lightgbm, xgboost, catboost)
- Portfolio constraints (max single-stock weight, sector caps)
- Rebalancing thresholds
- Event processing parameters

**API Keys**: See [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md) for:
- Which API keys you need
- How to get free API keys
- Recommended setup for different use cases
- How to use mock providers (no keys needed)

API keys should be set in `.env` file. The system defaults to mock providers so it works immediately without any API keys.

## Example Usage

### Python API

```python
from stock_engine.models import MarketModel
from stock_engine.portfolio import PortfolioOptimizer, TradeGenerator
from stock_engine.data import get_user_data_store
from datetime import date

# Load trained model
model = MarketModel(model_type='lightgbm')
model.load('models/saved/market_model_lightgbm_2023-12-31.pkl')

# Initialize components
optimizer = PortfolioOptimizer(market_model=model)
trade_generator = TradeGenerator(optimizer)

# Generate recommendations
trades = trade_generator.generate_trades(
    user_id=1,
    tickers=['AAPL', 'GOOGL', 'MSFT'],
    portfolio_value=100000.0,
    as_of_date=date.today()
)

print(trades)
```

### REST API

```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "portfolio_value": 100000.0,
    "tickers": ["AAPL", "GOOGL", "MSFT"]
  }'

# Get user positions
curl "http://localhost:8000/users/1/positions"
```

## Backtesting

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
    tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    initial_capital=100000.0,
    rebalance_frequency_days=30
)

# Compute metrics
metrics = backtester.compute_metrics(results)
print(f"CAGR: {metrics['portfolio_cagr']:.2%}")
print(f"Sharpe: {metrics['portfolio_sharpe']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

## Data Sources

### Market Data
- **Polygon.io** (production): Real-time and historical market data
- **Alpha Vantage**: Alternative provider
- **Mock Provider**: For testing/development

### News Data
- **Finnhub**: Financial news and sentiment
- **NewsAPI**: General news
- **Mock Provider**: For testing/development

### API Keys

**Important**: The system works with **mock providers** by default - no API keys needed for testing!

For production use with real data, see **[API_KEYS_GUIDE.md](API_KEYS_GUIDE.md)** for:
- Which API keys to get (free options available)
- Step-by-step setup instructions
- Recommended providers for different use cases
- Cost estimates

Quick summary: Set API keys in `.env` file:
- `MARKET_DATA_API_KEY`: Market data provider API key (optional)
- `NEWS_API_KEY`: News provider API key (optional)
- `OPENAI_API_KEY`: (Optional) For LLM-based event classification

## Development

### Project Structure

```
Stock/
├── config/                 # Configuration files
├── scripts/                # Training and inference scripts
├── stock_engine/           # Main package
│   ├── data/              # Data access layer
│   ├── features/          # Feature engineering
│   ├── events/            # Event processing
│   ├── models/            # ML models
│   ├── portfolio/         # Portfolio optimization
│   ├── backtesting/       # Backtesting framework
│   └── api/               # API server
├── models/saved/          # Trained models
├── data/                  # Database files
└── logs/                  # Log files
```

### Adding New Data Providers

1. Implement provider class in `stock_engine/data/`:
   - Inherit from `MarketDataProvider` or `NewsProvider`
   - Implement required methods
2. Add provider factory logic in `get_market_data_provider()` or `get_news_provider()`
3. Update `config/config.yaml` with new provider name

### Adding New Features

1. Add feature computation to `StockFeatureEngine.compute_features()` or `UserFeatureEngine.compute_features()`
2. Ensure feature is included in training data preparation
3. Update model training scripts if needed

## Testing

```bash
# Run tests (when implemented)
pytest tests/
```

## Production Deployment

### TODO: Production Checklist

- [ ] Replace mock providers with real APIs
- [ ] Set up PostgreSQL/MySQL database
- [ ] Implement authentication for API
- [ ] Add caching layer (Redis)
- [ ] Set up monitoring and logging
- [ ] Deploy with Docker/Kubernetes
- [ ] Set up CI/CD pipeline
- [ ] Implement model versioning
- [ ] Add comprehensive error handling

See [ARCHITECTURE.md](ARCHITECTURE.md) for more details on production considerations.

## License

[Add license information]

## Contributing

[Add contributing guidelines]

## Acknowledgments

[Add acknowledgments]
