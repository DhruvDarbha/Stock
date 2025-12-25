"""User and portfolio data store."""

import sqlite3
import pandas as pd
from datetime import date, datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..config import get_config


class UserDataStore:
    """Database abstraction for user and portfolio data."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize user data store.
        
        Args:
            db_path: Path to SQLite database file
        """
        config = get_config()
        self.db_path = db_path or config.db_path
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                risk_profile TEXT NOT NULL,  -- conservative, moderate, aggressive
                investment_horizon TEXT NOT NULL,  -- short, medium, long
                constraints_json TEXT,  -- JSON: max_single_stock_weight, sector_caps, excluded_industries
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Accounts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                broker_name TEXT,
                broker_account_id TEXT,
                connection_status TEXT DEFAULT 'disconnected',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, broker_name, broker_account_id)
            )
        """)
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                shares REAL NOT NULL,
                cost_basis REAL NOT NULL,
                acquired_date DATE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, ticker)
            )
        """)
        
        # Recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,  -- buy, sell, hold
                shares REAL,
                weight REAL,  -- Target portfolio weight
                rationale TEXT,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # User actions table (what users actually did)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                ticker TEXT NOT NULL,
                action_taken TEXT NOT NULL,  -- buy, sell, hold
                shares REAL,
                price REAL,
                source_recommendation_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (source_recommendation_id) REFERENCES recommendations(id)
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_user_ticker ON positions(user_id, ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_user_timestamp ON recommendations(user_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_actions_user_timestamp ON user_actions(user_id, timestamp)")
        
        conn.commit()
        conn.close()
    
    # User methods
    def create_user(
        self,
        username: str,
        risk_profile: str = 'moderate',
        investment_horizon: str = 'medium',
        constraints: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Create a new user.
        
        Returns:
            User ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        constraints_json = json.dumps(constraints or {})
        
        cursor.execute("""
            INSERT INTO users (username, risk_profile, investment_horizon, constraints_json)
            VALUES (?, ?, ?, ?)
        """, (username, risk_profile, investment_horizon, constraints_json))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        user = dict(row)
        user['constraints'] = json.loads(user.get('constraints_json') or '{}')
        del user['constraints_json']
        return user
    
    # Position methods
    def get_positions(self, user_id: int) -> pd.DataFrame:
        """Get all positions for a user."""
        conn = self._get_connection()
        
        df = pd.read_sql_query(
            "SELECT * FROM positions WHERE user_id = ?",
            conn,
            params=(user_id,)
        )
        
        conn.close()
        return df
    
    def update_position(
        self,
        user_id: int,
        ticker: str,
        shares: float,
        cost_basis: float,
        acquired_date: Optional[date] = None
    ):
        """Update or create a position."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        acquired_date_str = acquired_date.isoformat() if acquired_date else None
        
        cursor.execute("""
            INSERT INTO positions (user_id, ticker, shares, cost_basis, acquired_date)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, ticker) DO UPDATE SET
                shares = excluded.shares,
                cost_basis = excluded.cost_basis,
                acquired_date = excluded.acquired_date,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, ticker, shares, cost_basis, acquired_date_str))
        
        conn.commit()
        conn.close()
    
    # Recommendation methods
    def create_recommendation(
        self,
        user_id: int,
        ticker: str,
        action: str,
        shares: Optional[float] = None,
        weight: Optional[float] = None,
        rationale: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> int:
        """Create a recommendation."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO recommendations (user_id, timestamp, ticker, action, shares, weight, rationale, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            datetime.now(),
            ticker,
            action,
            shares,
            weight,
            rationale,
            model_version
        ))
        
        rec_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return rec_id
    
    def get_recommendations(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get recommendations for a user."""
        conn = self._get_connection()
        
        query = "SELECT * FROM recommendations WHERE user_id = ?"
        params = [user_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    # User action methods
    def record_user_action(
        self,
        user_id: int,
        ticker: str,
        action_taken: str,
        shares: float,
        price: float,
        source_recommendation_id: Optional[int] = None
    ) -> int:
        """Record a user action (what they actually did)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_actions (user_id, timestamp, ticker, action_taken, shares, price, source_recommendation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            datetime.now(),
            ticker,
            action_taken,
            shares,
            price,
            source_recommendation_id
        ))
        
        action_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return action_id
    
    def get_user_actions(
        self,
        user_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get user action history."""
        conn = self._get_connection()
        
        query = "SELECT * FROM user_actions WHERE user_id = ?"
        params = [user_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    # Behavior analysis methods
    def get_user_behavior_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Compute behavioral statistics for a user.
        
        Returns:
            Dictionary with behavior metrics
        """
        conn = self._get_connection()
        
        # Get recommendations and actions
        recommendations = self.get_recommendations(user_id)
        actions = self.get_user_actions(user_id)
        
        if recommendations.empty:
            return {
                'total_recommendations': 0,
                'follow_rate': 0.0,
                'avg_holding_period_days': 0.0,
                'max_drawdown_tolerated': 0.0,
                'volatility_sensitivity': 0.0
            }
        
        # Merge recommendations with actions
        merged = recommendations.merge(
            actions,
            left_on='id',
            right_on='source_recommendation_id',
            how='left',
            suffixes=('_rec', '_action')
        )
        
        # Calculate follow rate
        followed = merged[merged['action_taken'].notna()]
        total_recs = len(recommendations)
        follow_rate = len(followed) / total_recs if total_recs > 0 else 0.0
        
        # Calculate average holding period (simplified - would need more detailed tracking)
        # This is a placeholder - would need position history tracking
        
        # Max drawdown tolerated (placeholder - would need portfolio value history)
        
        stats = {
            'total_recommendations': total_recs,
            'follow_rate': follow_rate,
            'avg_holding_period_days': 0.0,  # TODO: Implement proper calculation
            'max_drawdown_tolerated': 0.0,  # TODO: Implement proper calculation
            'volatility_sensitivity': 0.0  # TODO: Implement proper calculation
        }
        
        conn.close()
        return stats


def get_user_data_store(db_path: Optional[str] = None) -> UserDataStore:
    """
    Factory function to get user data store instance.
    
    Returns:
        UserDataStore instance
    """
    return UserDataStore(db_path)

