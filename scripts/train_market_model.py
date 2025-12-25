"""Script to train the market prediction model."""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_engine.models import MarketModel
from stock_engine.config import get_config


def main():
    """Train market model on historical data."""
    config = get_config()
    
    # Get training parameters
    train_start_str = config.get('training.train_start_date', '2020-01-01')
    train_end_str = config.get('training.train_end_date', '2023-12-31')
    
    train_start = datetime.strptime(train_start_str, '%Y-%m-%d').date()
    train_end = datetime.strptime(train_end_str, '%Y-%m-%d').date()
    
    # Check provider and warn if needed
    provider = config.market_data_provider
    if provider != 'mock':
        print(f"⚠️  Using {provider} provider - may have rate limits or data restrictions")
    
    # TODO: Load ticker universe from config or file
    # Using smaller list to avoid rate limits with free tier
    # Polygon free tier: ~5 requests/minute
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD'
    ]
    
    provider = config.market_data_provider
    if provider == 'mock':
        print("ℹ️  Using mock data provider - instant training with synthetic data")
    else:
        print("⚠️  Note: Using reduced ticker list to avoid rate limits.")
        print(f"⚠️  {provider} may have rate limits. Training will take time.")
    
    print(f"Training market model from {train_start} to {train_end}")
    print(f"Tickers: {len(tickers)}")
    
    # Initialize model
    model_type = config.get('models.market_model.type', 'lightgbm')
    model = MarketModel(model_type=model_type)
    
    print("\nPreparing training data...")
    # Prepare training data
    X, y = model.prepare_training_data(tickers, train_start, train_end)
    
    print(f"Training samples: {len(X)}")
    print(f"Features: {len(X.columns)}")
    print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")
    
    # Train model
    print("\nTraining model...")
    validation_split = config.get('training.validation_split', 0.2)
    model.train(X, y, validation_split=validation_split)
    
    # Save model
    model_dir = Path('models/saved')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'market_model_{model_type}_{train_end.isoformat()}.pkl'
    model.save(str(model_path))
    
    print(f"\nModel saved to: {model_path}")
    print("Training complete!")


if __name__ == '__main__':
    main()

