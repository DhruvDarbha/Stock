"""Example usage of the stock recommendation engine."""

import sys
from pathlib import Path
from datetime import date

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_engine.data import get_user_data_store, get_market_data_provider
from stock_engine.features import StockFeatureEngine
from stock_engine.models import MarketModel
from stock_engine.portfolio import PortfolioOptimizer, TradeGenerator
from stock_engine.events import EventEngine


def example_1_basic_usage():
    """Example 1: Basic feature computation and prediction."""
    print("=" * 80)
    print("Example 1: Computing stock features")
    print("=" * 80)
    
    # Initialize feature engine
    feature_engine = StockFeatureEngine()
    
    # Compute features for a stock
    ticker = "AAPL"
    as_of_date = date.today()
    
    print(f"\nComputing features for {ticker} as of {as_of_date}...")
    features = feature_engine.compute_features(ticker, as_of_date)
    
    print(f"\nSample features:")
    for key, value in list(features.items())[:10]:
        if not (isinstance(value, float) and value != value):  # Skip NaN
            print(f"  {key}: {value}")
    
    print("\n✓ Example 1 complete\n")


def example_2_user_management():
    """Example 2: User and portfolio management."""
    print("=" * 80)
    print("Example 2: User and portfolio management")
    print("=" * 80)
    
    # Get user data store
    user_store = get_user_data_store()
    
    # Create a user
    print("\nCreating test user...")
    user_id = user_store.create_user(
        username="example_user",
        risk_profile="moderate",
        investment_horizon="medium",
        constraints={
            "max_single_stock_weight": 0.10,
            "max_sector_weight": 0.30
        }
    )
    print(f"Created user with ID: {user_id}")
    
    # Add some positions
    print("\nAdding positions...")
    user_store.update_position(user_id, "AAPL", 10, 150.0)
    user_store.update_position(user_id, "GOOGL", 5, 2800.0)
    print("Added positions for AAPL and GOOGL")
    
    # Get positions
    positions = user_store.get_positions(user_id)
    print(f"\nCurrent positions:")
    print(positions[['ticker', 'shares', 'cost_basis']])
    
    # Get user info
    user = user_store.get_user(user_id)
    print(f"\nUser profile:")
    print(f"  Risk profile: {user['risk_profile']}")
    print(f"  Investment horizon: {user['investment_horizon']}")
    
    print("\n✓ Example 2 complete\n")


def example_3_event_processing():
    """Example 3: Event processing and classification."""
    print("=" * 80)
    print("Example 3: Event processing")
    print("=" * 80)
    
    # Initialize event engine
    event_engine = EventEngine()
    
    # Process events for a ticker
    ticker = "AAPL"
    as_of_date = date.today()
    
    print(f"\nProcessing events for {ticker}...")
    event_features = event_engine.process_ticker_events(
        ticker,
        as_of_date,
        lookback_days=30
    )
    
    print(f"\nEvent features:")
    for key, value in event_features.items():
        print(f"  {key}: {value}")
    
    # Check if risk multiplier should be applied
    should_apply, multiplier = event_engine.should_apply_risk_multiplier(
        ticker, as_of_date
    )
    print(f"\nRisk multiplier check:")
    print(f"  Should apply: {should_apply}")
    print(f"  Multiplier: {multiplier}")
    
    print("\n✓ Example 3 complete\n")


def example_4_model_prediction():
    """Example 4: Model prediction (requires trained model)."""
    print("=" * 80)
    print("Example 4: Model prediction")
    print("=" * 80)
    
    model_dir = Path("models/saved")
    model_files = list(model_dir.glob("market_model_*.pkl"))
    
    if not model_files:
        print("\n⚠ No trained model found.")
        print("Train a model first: python scripts/train_market_model.py")
        return
    
    # Load model
    model_path = model_files[0]
    print(f"\nLoading model: {model_path.name}")
    
    model = MarketModel(model_type="lightgbm")
    model.load(str(model_path))
    
    # Compute features for prediction
    feature_engine = StockFeatureEngine()
    tickers = ["AAPL", "GOOGL", "MSFT"]
    
    print(f"\nComputing features for {len(tickers)} tickers...")
    features_df = feature_engine.compute_features_batch(tickers, date.today())
    
    if features_df.empty:
        print("⚠ Could not compute features. Using mock data provider may have limitations.")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(features_df)
    risk_scores = model.predict_risk(features_df)
    
    print(f"\nPredictions (excess returns):")
    for ticker in tickers:
        if ticker in predictions.index:
            print(f"  {ticker}: {predictions[ticker]:.4f} (risk: {risk_scores[ticker]:.4f})")
    
    print("\n✓ Example 4 complete\n")


def example_5_full_pipeline():
    """Example 5: Full recommendation pipeline (requires trained model)."""
    print("=" * 80)
    print("Example 5: Full recommendation pipeline")
    print("=" * 80)
    
    model_dir = Path("models/saved")
    model_files = list(model_dir.glob("market_model_*.pkl"))
    
    if not model_files:
        print("\n⚠ No trained model found.")
        print("Train a model first: python scripts/train_market_model.py")
        return
    
    # Load model and initialize components
    model_path = model_files[0]
    print(f"\nLoading model: {model_path.name}")
    
    market_model = MarketModel(model_type="lightgbm")
    market_model.load(str(model_path))
    
    event_engine = EventEngine()
    optimizer = PortfolioOptimizer(market_model, event_engine=event_engine)
    trade_generator = TradeGenerator(optimizer)
    
    # Get or create user
    user_store = get_user_data_store()
    user_id = 1
    user = user_store.get_user(user_id)
    
    if not user:
        user_id = user_store.create_user(
            username="demo_user",
            risk_profile="moderate",
            investment_horizon="medium"
        )
        print(f"Created user ID: {user_id}")
    
    # Generate recommendations
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    portfolio_value = 100000.0
    
    print(f"\nGenerating recommendations for {len(tickers)} tickers...")
    trades = trade_generator.generate_trades(
        user_id=user_id,
        tickers=tickers,
        portfolio_value=portfolio_value,
        as_of_date=date.today()
    )
    
    if not trades.empty:
        print(f"\nRecommendations:")
        for _, row in trades.iterrows():
            if row['action'] != 'hold':
                print(f"\n  {row['action'].upper()}: {row['shares']} shares of {row['ticker']}")
                print(f"    Current weight: {row['current_weight']:.2%}")
                print(f"    Target weight: {row['target_weight']:.2%}")
                print(f"    Rationale: {row['rationale']}")
        print("\n✓ Example 5 complete\n")
    else:
        print("\n⚠ No trades generated. This may be normal if rebalance threshold not met.")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Stock Recommendation Engine - Example Usage")
    print("=" * 80)
    
    try:
        example_1_basic_usage()
        example_2_user_management()
        example_3_event_processing()
        example_4_model_prediction()
        example_5_full_pipeline()
        
        print("=" * 80)
        print("All examples complete!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Train models: python scripts/train_market_model.py")
        print("2. Generate recommendations: python scripts/generate_recommendations.py")
        print("3. Run API server: uvicorn stock_engine.api.server:app --host 0.0.0.0 --port 8000")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

