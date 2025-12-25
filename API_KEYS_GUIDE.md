# API Keys Guide

This guide explains which API keys you need and how to get them.

## Quick Start: Minimal Setup (Free Options)

For **testing and development**, you can use the **mock providers** which don't require any API keys. The system will generate synthetic data.

However, for **real predictions and production use**, you'll need at least one API key.

## Required API Keys

### 1. Market Data Provider (Choose ONE)

You need **at least one** of these for real stock price and fundamental data:

#### Option A: Polygon.io (Recommended)
- **Free Tier**: Limited requests/month
- **Paid Plans**: Start at $29/month
- **Why Choose**: Professional-grade, comprehensive data
- **Get API Key**: https://polygon.io/dashboard/signup
- **Best For**: Production use

#### Option B: Alpha Vantage (Free)
- **Free Tier**: 5 API calls/minute, 500 calls/day
- **Paid Plans**: Available
- **Why Choose**: Completely free tier
- **Get API Key**: https://www.alphavantage.co/support/#api-key
- **Best For**: Development and testing

#### Option C: Finnhub (Free)
- **Free Tier**: 60 calls/minute
- **Paid Plans**: Available
- **Why Choose**: Includes both market data AND news
- **Get API Key**: https://finnhub.io/register
- **Best For**: Getting both market data and news with one key

#### Option D: Tiingo
- **Free Tier**: Limited
- **Paid Plans**: Available
- **Get API Key**: https://api.tiingo.com/
- **Best For**: Alternative option

### 2. News Provider (Choose ONE)

You need **at least one** of these for news and sentiment data:

#### Option A: Finnhub (Recommended if already using for market data)
- **Same API key** as market data
- Includes financial news with sentiment

#### Option B: NewsAPI (Free)
- **Free Tier**: 100 requests/day
- **Paid Plans**: Available
- **Get API Key**: https://newsapi.org/register
- **Note**: General news, may need filtering for financial news

#### Option C: FinancialModelingPrep
- **Free Tier**: Limited
- **Get API Key**: https://site.financialmodelingprep.com/developer/docs/
- **Best For**: Financial-specific news

## Optional API Keys

### OpenAI API Key (Optional but Recommended)
- **Purpose**: Enhanced event classification using LLM
- **Why**: Better extraction of events (endorsements, controversies, etc.)
- **Without It**: System uses rule-based classification (still works!)
- **Get API Key**: https://platform.openai.com/api-keys
- **Cost**: Pay-per-use, typically $0.01-0.10 per 1000 requests

### Social Media APIs (Future Enhancement)
- Twitter/X API: For social sentiment
- Reddit API: For Reddit mentions
- **Status**: Not yet implemented, but structure is in place

## Recommended Setup by Use Case

### For Development/Testing (Free)
```env
# Use mock providers (no API keys needed)
# Just leave all keys empty or don't set them
# The system will automatically use mock providers
```

### For Production - Budget Conscious
```env
# Use Alpha Vantage (free) for market data
MARKET_DATA_API_KEY=your_alphavantage_key
# In config.yaml, set: provider: "alpha_vantage"

# Use NewsAPI (free) for news
NEWS_API_KEY=your_newsapi_key
# In config.yaml, set: provider: "newsapi"
```

### For Production - Best Quality
```env
# Use Polygon.io for market data
MARKET_DATA_API_KEY=your_polygon_key
# In config.yaml, set: provider: "polygon"

# Use Finnhub for news (or same Polygon if it includes news)
NEWS_API_KEY=your_finnhub_key
# In config.yaml, set: provider: "finnhub"

# Optional: OpenAI for better event extraction
OPENAI_API_KEY=your_openai_key
```

### For Production - Single Provider
```env
# Use Finnhub for both market data AND news
MARKET_DATA_API_KEY=your_finnhub_key
NEWS_API_KEY=your_finnhub_key  # Same key!
# In config.yaml:
#   data_sources.market_data.provider: "finnhub"
#   data_sources.news.provider: "finnhub"
```

## Configuration

After getting your API keys:

1. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** and add your API keys:
   ```env
   MARKET_DATA_API_KEY=your_actual_key_here
   NEWS_API_KEY=your_actual_key_here
   ```

3. **Update `config/config.yaml`** to specify which providers to use:
   ```yaml
   data_sources:
     market_data:
       provider: "polygon"  # or "alpha_vantage", "finnhub", etc.
     news:
       provider: "finnhub"  # or "newsapi", etc.
   ```

## Testing Without API Keys

The system includes **mock providers** that generate synthetic data. To use them:

1. **Leave API keys empty** in `.env`, or
2. **Set providers to "mock"** in `config/config.yaml`:
   ```yaml
   data_sources:
     market_data:
       provider: "mock"
     news:
       provider: "mock"
   ```

This allows you to:
- Test the entire pipeline
- Develop features
- Run examples
- Train models (with synthetic data)

**Note**: Predictions will not be meaningful with mock data, but the system structure will work.

## Verifying API Keys

After setting up your keys, test them:

```python
from stock_engine.data import get_market_data_provider, get_news_provider

# Test market data provider
market_data = get_market_data_provider()
if market_data.health_check():
    print("✓ Market data provider is working!")
else:
    print("✗ Market data provider failed - check your API key")

# Test news provider
news_data = get_news_provider()
if news_data.health_check():
    print("✓ News provider is working!")
else:
    print("✗ News provider failed - check your API key")
```

Or run the example script:
```bash
python scripts/example_usage.py
```

## Cost Estimates

### Free Tier Only
- **Alpha Vantage**: $0 (limited to 500 calls/day)
- **NewsAPI**: $0 (limited to 100 calls/day)
- **Total**: **$0/month**

### Basic Production
- **Polygon.io Starter**: ~$29/month
- **NewsAPI Developer**: ~$449/month (or use Finnhub free tier)
- **Total**: ~$29-449/month

### Recommended Production
- **Polygon.io Starter**: ~$29/month
- **Finnhub Free/Starter**: $0-59/month
- **OpenAI**: ~$10-50/month (depending on usage)
- **Total**: ~$39-138/month

## Troubleshooting

### "API key not set" warnings
- This is normal if using mock providers
- The system will automatically fall back to mock data

### Rate limit errors
- Check your API provider's rate limits
- Adjust rate limiting in `config/config.yaml`
- Consider upgrading to a paid plan

### "Provider not supported" error
- Check that the provider name in `config.yaml` matches exactly
- Make sure you've implemented the provider (currently: polygon, finnhub, mock)
- See `stock_engine/data/market_data.py` and `news_data.py` for supported providers

## Next Steps

1. **Start with mock providers** to test the system
2. **Get free API keys** (Alpha Vantage, NewsAPI) for real data
3. **Upgrade as needed** for production use

The system is designed to work seamlessly with mock data, so you can develop and test without any API keys!

