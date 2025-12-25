# Quick Start Guide - How to Run the Code

## ✅ Prerequisites
1. Python 3.8+ installed
2. API keys configured (already done!):
   - ✅ Polygon API key (market data)
   - ✅ NewsAPI key (news data)
   - ✅ OpenAI API key (event classification)

## 📦 Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Step 2: Test the System

Run the example script to test everything works:

```bash
python scripts/example_usage.py
```

This will:
- Test data connections (Polygon, NewsAPI)
- Create a test user
- Compute features for stocks
- Show event processing
- Test the full pipeline

## 🎯 Step 3: Run Specific Tasks

### Generate Recommendations for a User

```bash
python scripts/generate_recommendations.py --user-id 1 --portfolio-value 100000
```

This will:
- Generate buy/sell/hold recommendations
- Show rationale for each trade
- Save recommendations to database

**Note**: You need a trained model first (see Step 4)

### Train Market Prediction Model

```bash
python scripts/train_market_model.py
```

This will:
- Fetch historical stock data from Polygon
- Compute features
- Train a LightGBM model
- Save model to `models/saved/`

**Time**: Takes 10-30 minutes depending on data size

### Train Behavior Model

```bash
python scripts/train_behavior_model.py
```

**Note**: Requires user recommendation/action history in database

## 🌐 Step 4: Run the API Server

Start the REST API server:

```bash
uvicorn stock_engine.api.server:app --host 0.0.0.0 --port 8000
```

Then visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### API Usage Examples

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

## 📊 Step 5: Backtest Strategy

```python
from stock_engine.models import MarketModel
from stock_engine.backtesting import Backtester
from datetime import date
from pathlib import Path

# Load trained model
model = MarketModel(model_type='lightgbm')
model_path = Path('models/saved')
model_files = list(model_path.glob('market_model_*.pkl'))
if model_files:
    model.load(str(model_files[0]))
    
    # Run backtest
    backtester = Backtester(model)
    results = backtester.run_backtest(
        tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=100000.0
    )
    
    # Get metrics
    metrics = backtester.compute_metrics(results)
    print(f"CAGR: {metrics['portfolio_cagr']:.2%}")
    print(f"Sharpe: {metrics['portfolio_sharpe']:.2f}")
```

## 🔧 Common Issues

### "Module not found" error
```bash
# Make sure you're in the project root directory
cd /Users/dhruv/Documents/GitHub/Stock

# Install dependencies
pip install -r requirements.txt
```

### "Model not found" error
```bash
# Train the model first
python scripts/train_market_model.py
```

### "API key not set" warning
- Check `.env` file exists and has your keys
- Verify `config/config.yaml` has correct provider names
- Make sure you're using the right environment variable names

### Rate limit errors
- NewsAPI free tier: 100 requests/day
- Reduce frequency of API calls
- Consider upgrading to paid tier

## 🎓 Learning Path

1. **Start Simple**: Run `example_usage.py` to see how everything works
2. **Train Model**: Train a market model with historical data
3. **Generate Recs**: Create recommendations for a user
4. **Explore API**: Use the REST API to integrate with other systems
5. **Backtest**: Evaluate strategy performance
6. **Iterate**: Adjust models, features, and optimization

## 📚 Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [API_KEYS_GUIDE.md](API_KEYS_GUIDE.md) for API key details
- Check `config/config.yaml` for all configuration options

Happy coding! 🚀
