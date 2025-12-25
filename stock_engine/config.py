"""Configuration management for the stock recommendation engine."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file."""
        config_file = Path(__file__).parent.parent / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Replace environment variable placeholders
        self._substitute_env_vars(self.config)
    
    def _substitute_env_vars(self, obj: Any) -> None:
        """Recursively substitute environment variables in config."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    obj[key] = os.getenv(env_var, value)
                else:
                    self._substitute_env_vars(value)
        elif isinstance(obj, list):
            for item in obj:
                self._substitute_env_vars(item)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example: config.get('data_sources.market_data.provider')
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    @property
    def market_data_provider(self) -> str:
        """Get market data provider name."""
        return self.get('data_sources.market_data.provider', 'mock')
    
    @property
    def news_provider(self) -> str:
        """Get news provider name."""
        return self.get('data_sources.news.provider', 'mock')
    
    @property
    def db_path(self) -> str:
        """Get database path."""
        return self.get('database.path', 'data/stock_engine.db')
    
    @property
    def benchmark_ticker(self) -> str:
        """Get benchmark ticker."""
        return self.get('portfolio.benchmark_ticker', 'SPY')


# Global config instance
_config: Config = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

