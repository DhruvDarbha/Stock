#!/usr/bin/env python3
"""Simple test to verify API connections and basic functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Stock Recommendation Engine Setup...")
print("=" * 60)

# Test 1: Configuration
print("\n1. Testing configuration...")
try:
    from stock_engine.config import get_config
    config = get_config()
    print(f"   ✓ Config loaded")
    print(f"   - Market data provider: {config.market_data_provider}")
    print(f"   - News provider: {config.news_provider}")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test 2: Market Data Provider
print("\n2. Testing market data provider...")
try:
    from stock_engine.data import get_market_data_provider
    market_data = get_market_data_provider()
    print(f"   ✓ Market data provider initialized: {type(market_data).__name__}")
    
    # Quick health check
    if market_data.health_check():
        print(f"   ✓ Health check passed")
    else:
        print(f"   ⚠ Health check failed (may need API key)")
except Exception as e:
    print(f"   ✗ Market data error: {e}")

# Test 3: News Provider
print("\n3. Testing news provider...")
try:
    from stock_engine.data import get_news_provider
    news_data = get_news_provider()
    print(f"   ✓ News provider initialized: {type(news_data).__name__}")
    
    # Quick health check
    if news_data.health_check():
        print(f"   ✓ Health check passed")
    else:
        print(f"   ⚠ Health check failed (may need API key)")
except Exception as e:
    print(f"   ✗ News data error: {e}")

# Test 4: User Data Store
print("\n4. Testing user data store...")
try:
    from stock_engine.data import get_user_data_store
    user_store = get_user_data_store()
    print(f"   ✓ User data store initialized")
    
    # Create a test user
    user_id = user_store.create_user(
        username="test_user",
        risk_profile="moderate",
        investment_horizon="medium"
    )
    print(f"   ✓ Test user created (ID: {user_id})")
except Exception as e:
    print(f"   ✗ User data store error: {e}")

# Test 5: Feature Engineering (basic)
print("\n5. Testing feature engineering...")
try:
    from stock_engine.features import StockFeatureEngine
    feature_engine = StockFeatureEngine()
    print(f"   ✓ Feature engine initialized")
except Exception as e:
    print(f"   ✗ Feature engineering error: {e}")

# Test 6: Event Processing
print("\n6. Testing event processing...")
try:
    from stock_engine.events import EventEngine
    event_engine = EventEngine()
    print(f"   ✓ Event engine initialized")
except Exception as e:
    print(f"   ✗ Event processing error: {e}")

print("\n" + "=" * 60)
print("Setup test complete!")
print("\nNext steps:")
print("  - If LightGBM/XGBoost errors: Install libomp via Homebrew:")
print("    brew install libomp")
print("  - Or skip model training for now and test other components")
print("  - Run: python3 scripts/train_market_model.py (after fixing LightGBM)")
