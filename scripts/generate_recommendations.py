"""Script to generate recommendations for a user."""

import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_engine.models import MarketModel, BehaviorModel
from stock_engine.portfolio import PortfolioOptimizer, TradeGenerator
from stock_engine.events import EventEngine
from stock_engine.config import get_config
from stock_engine.data import get_user_data_store


def main(user_id: int = 1, portfolio_value: float = 100000.0):
    """
    Generate recommendations for a user.
    
    Args:
        user_id: User ID
        portfolio_value: Total portfolio value in dollars
    """
    config = get_config()
    
    print(f"Generating recommendations for user {user_id}")
    print(f"Portfolio value: ${portfolio_value:,.2f}")
    
    # Load models
    model_dir = Path('models/saved')
    model_type = config.get('models.market_model.type', 'lightgbm')
    
    # Try to load market model
    market_model_path = None
    for path in model_dir.glob(f'market_model_{model_type}_*.pkl'):
        market_model_path = path
        break
    
    if not market_model_path or not market_model_path.exists():
        print("\nMarket model not found. Please train the model first:")
        print("  python scripts/train_market_model.py")
        return
    
    market_model = MarketModel(model_type=model_type)
    market_model.load(str(market_model_path))
    print(f"Loaded market model: {market_model_path.name}")
    
    # Try to load behavior model (optional)
    behavior_model = None
    behavior_model_path = model_dir / f'behavior_model_{model_type}_follow.pkl'
    if behavior_model_path.exists():
        behavior_model = BehaviorModel(model_type=model_type)
        behavior_model.load(str(behavior_model_path))
        print(f"Loaded behavior model: {behavior_model_path.name}")
    
    # Initialize components
    event_engine = EventEngine()
    optimizer = PortfolioOptimizer(
        market_model=market_model,
        behavior_model=behavior_model,
        event_engine=event_engine
    )
    trade_generator = TradeGenerator(optimizer)
    
    # Get user
    user_store = get_user_data_store()
    user = user_store.get_user(user_id)
    
    if not user:
        print(f"\nUser {user_id} not found. Creating test user...")
        user_id = user_store.create_user(
            username=f"test_user_{user_id}",
            risk_profile='moderate',
            investment_horizon='medium'
        )
        print(f"Created user ID: {user_id}")
    
    # TODO: Get ticker universe from config or user preferences
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD'
    ]
    
    print(f"\nAnalyzing {len(tickers)} tickers...")
    
    # Generate trades
    as_of_date = date.today()
    trades_df = trade_generator.generate_trades(
        user_id=user_id,
        tickers=tickers,
        portfolio_value=portfolio_value,
        as_of_date=as_of_date
    )
    
    if trades_df.empty:
        print("\nNo trades generated.")
        return
    
    # Display predictions and recommendations
    print("\n" + "="*80)
    print("STOCK PREDICTIONS & RECOMMENDATIONS")
    print("="*80)
    
    # Get current prices for display
    from stock_engine.data import get_market_data_provider
    market_data = get_market_data_provider()
    current_prices = {}
    for ticker in tickers:
        try:
            price_df = market_data.get_ohlcv(ticker, as_of_date - timedelta(days=5), as_of_date)
            if not price_df.empty:
                current_prices[ticker] = price_df.iloc[-1]['close']
            else:
                current_prices[ticker] = None
        except:
            current_prices[ticker] = None
    
    # First show all stocks with predictions
    print("\n📊 PREDICTIONS FOR ALL STOCKS:")
    print("=" * 100)
    print(f"{'Ticker':<8} {'Price':<10} {'Exp Return':<13} {'Risk':<8} {'Action':<8} {'Shares':<8} {'Amount':<12} {'Target %':<10}")
    print("-" * 100)
    
    for _, row in trades_df.iterrows():
        ticker = row['ticker']
        price = current_prices.get(ticker, row.get('current_price', 0))
        pred_return = row.get('expected_excess_return', 0.0)
        risk = row.get('risk_score', 0.15)
        action = row['action'].upper()
        shares = int(row['shares']) if row['action'] != 'hold' else 0
        target_weight = row['target_weight']
        amount = shares * price if price and shares > 0 else 0
        
        # Format prediction - default to 0.00% if missing
        if pd.isna(pred_return):
            pred_return = 0.0
        if pd.isna(risk):
            risk = 0.15
        if price is None or price == 0:
            price = 0.0
        
        return_str = f"{pred_return:>7.2%}"
        risk_str = f"{risk:>6.2f}"
        price_str = f"${price:>7.2f}" if price > 0 else "     N/A"
        amount_str = f"${amount:>10,.0f}" if amount > 0 else "         -"
        
        # Color coding
        if action == 'BUY':
            action_display = f"🟢 {action}"
        elif action == 'SELL':
            action_display = f"🔴 {action}"
        else:
            action_display = f"⏸️  {action}"
        
        print(f"{ticker:<8} {price_str:<10} {return_str:<13} {risk_str:<8} {action_display:<10} {shares:<8} {amount_str:<12} {target_weight:>6.2%}")
    
    # Then show detailed recommendations
    print("\n" + "="*80)
    print("DETAILED RECOMMENDATIONS")
    print("="*80)
    
    buy_recs = trades_df[trades_df['action'] == 'buy']
    sell_recs = trades_df[trades_df['action'] == 'sell']
    hold_recs = trades_df[trades_df['action'] == 'hold']
    
    if not buy_recs.empty:
        print("\n🟢 BUY RECOMMENDATIONS:")
        print("-" * 100)
        for _, row in buy_recs.iterrows():
            ticker = row['ticker']
            price = current_prices.get(ticker, row.get('current_price', 0))
            shares = int(row['shares'])
            amount = shares * price if price else 0
            
            print(f"\n✅ BUY {shares} shares of {ticker} @ ${price:.2f} per share")
            print(f"   💰 Investment amount: ${amount:,.2f}")
            print(f"   📈 Expected excess return vs market: {row['expected_excess_return']:+.2%}")
            print(f"   ⚠️  Risk score: {row['risk_score']:.2f}")
            print(f"   📊 Current position: {row['current_weight']:.2%} of portfolio")
            print(f"   🎯 Target position: {row['target_weight']:.2%} of portfolio")
            print(f"   📝 Rationale: {row['rationale']}")
    
    if not sell_recs.empty:
        print("\n🔴 SELL RECOMMENDATIONS:")
        print("-" * 100)
        for _, row in sell_recs.iterrows():
            ticker = row['ticker']
            price = current_prices.get(ticker, row.get('current_price', 0))
            shares = int(row['shares'])
            proceeds = shares * price if price else 0
            
            print(f"\n❌ SELL {shares} shares of {ticker} @ ${price:.2f} per share")
            print(f"   💰 Proceeds: ${proceeds:,.2f}")
            print(f"   📉 Expected excess return vs market: {row['expected_excess_return']:+.2%}")
            print(f"   ⚠️  Risk score: {row['risk_score']:.2f}")
            print(f"   📊 Current position: {row['current_weight']:.2%} of portfolio")
            print(f"   🎯 Target position: {row['target_weight']:.2%} of portfolio")
            print(f"   📝 Rationale: {row['rationale']}")
    
    if not hold_recs.empty:
        print("\n⏸️  HOLD (No Action Needed):")
        print("-" * 80)
        for _, row in hold_recs.iterrows():
            if row['target_weight'] > 0:  # Only show if we have a position
                print(f"   • {row['ticker']}: {row['rationale']}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total BUY recommendations: {len(buy_recs)}")
    print(f"Total SELL recommendations: {len(sell_recs)}")
    print(f"Total HOLD positions: {len(hold_recs)}")
    
    if not buy_recs.empty:
        total_buy_value = 0.0
        for _, row in buy_recs.iterrows():
            ticker = row['ticker']
            price = current_prices.get(ticker, row.get('current_price', 0))
            shares = int(row['shares'])
            if price:
                total_buy_value += shares * price
        print(f"Total investment needed: ${total_buy_value:,.2f}")
    
    if not sell_recs.empty:
        total_sell_value = 0.0
        for _, row in sell_recs.iterrows():
            ticker = row['ticker']
            price = current_prices.get(ticker, row.get('current_price', 0))
            shares = int(row['shares'])
            if price:
                total_sell_value += shares * price
        print(f"Total proceeds from sales: ${total_sell_value:,.2f}")
    
    # Save recommendations
    model_version = f"v1.0_{as_of_date.isoformat()}"
    trade_generator.save_recommendations(user_id, trades_df, model_version)
    print(f"\n\nRecommendations saved to database (model version: {model_version})")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate stock recommendations')
    parser.add_argument('--user-id', type=int, default=1, help='User ID')
    parser.add_argument('--portfolio-value', type=float, default=100000.0, help='Portfolio value in dollars')
    
    args = parser.parse_args()
    main(args.user_id, args.portfolio_value)

