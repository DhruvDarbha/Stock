"""Script to train the user behavior model."""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_engine.models import BehaviorModel
from stock_engine.config import get_config
from stock_engine.data import get_user_data_store


def main():
    """Train behavior model on user history."""
    config = get_config()
    
    # Get all users
    user_store = get_user_data_store()
    
    # TODO: Get user IDs from database
    # For now, create some mock users with history
    # In production, this would query the database
    
    print("Preparing training data...")
    print("NOTE: This script requires existing user recommendation/action history.")
    print("For initial training, create mock data or use simulated user behavior.")
    
    # Initialize model
    model_type = config.get('models.behavior_model.type', 'lightgbm')
    model = BehaviorModel(model_type=model_type)
    
    # TODO: Get actual user IDs from database
    user_ids = []  # Replace with actual user IDs
    
    if not user_ids:
        print("\nNo users with history found. Skipping training.")
        print("Create users and generate recommendations first, then retrain.")
        return
    
    # Prepare training data for 'follow' task
    print(f"\nPreparing data for {len(user_ids)} users...")
    X, y = model.prepare_training_data(
        user_ids,
        task='follow'
    )
    
    if X.empty:
        print("No training data available. Need user recommendation/action history.")
        return
    
    print(f"Training samples: {len(X)}")
    print(f"Features: {len(X.columns)}")
    print(f"Follow rate: {y.mean():.4f}")
    
    # Train model
    print("\nTraining model...")
    validation_split = config.get('training.validation_split', 0.2)
    model.train(X, y, validation_split=validation_split, task='follow')
    
    # Save model
    model_dir = Path('models/saved')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'behavior_model_{model_type}_follow.pkl'
    model.save(str(model_path))
    
    print(f"\nModel saved to: {model_path}")
    print("Training complete!")


if __name__ == '__main__':
    main()

