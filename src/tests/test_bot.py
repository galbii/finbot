import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import json

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot import Bot
from strategy import Strategy

class MockDataManager:
    """Mock implementation of DataManager for testing"""
    
    def fetch_data(self, ticker, start_date=None, end_date=None, force_download=False):
        """Mock fetch_data that returns generated data"""
        # Generate date range
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        date_rng = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create mock OHLCV data
        np.random.seed(42)  # for reproducibility
        base_price = 100.0
        prices = []
        for i in range(len(date_rng)):
            # Add some random noise and a slight upward trend
            change = np.random.normal(0.05, 1.0)  # mean 0.05%, std 1%
            base_price *= (1 + change / 100)
            prices.append(base_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, size=len(date_rng))
        }, index=date_rng)
        
        return df
    
    def clear_cache(self):
        """Mock clear_cache method"""
        pass

class TestBot(unittest.TestCase):
    """Test cases for the Bot class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock data manager
        self.mock_data_manager = MockDataManager()
        
        # Create a bot for testing
        self.bot = Bot('AAPL', start_date='2022-01-01', end_date='2022-12-31')
        
        # Replace the bot's data_manager with our mock and refetch data
        self.bot.data_manager = self.mock_data_manager
        self.bot._fetch_historical_data()
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up bot resources
        if hasattr(self, 'bot'):
            self.bot.delete()
    
    def test_initialization(self):
        """Test bot initialization"""
        self.assertEqual(self.bot.ticker, 'AAPL')
        self.assertIsNotNone(self.bot.historical_data)
        self.assertIsNotNone(self.bot.best_strategy)
        self.assertEqual(len(self.bot.strategies_history), 1)
        self.assertEqual(self.bot.best_strategy.grammar_string, "BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70")
    
    def test_position_size_calculation(self):
        """Test position size calculation logic"""
        # Test fixed amount
        pos_sizing_fixed = {'type': 'FIXED', 'amount': 5000}
        size = self.bot._calculate_position_size(pos_sizing_fixed, 10000, 100)
        self.assertEqual(size, 50)  # 5000/100 = 50 shares
        
        # Test percentage
        pos_sizing_pct = {'type': 'PERCENTAGE', 'percentage': 50}
        size = self.bot._calculate_position_size(pos_sizing_pct, 10000, 100)
        self.assertEqual(size, 50)  # 50% of 10000 = 5000, 5000/100 = 50 shares
        
        # Test risk-based
        pos_sizing_risk = {'type': 'RISK_BASED', 'risk_percentage': 2}
        size = self.bot._calculate_position_size(pos_sizing_risk, 10000, 100)
        self.assertEqual(size, 2)  # 2% of 10000 = 200, 200/100 = 2 shares
        
        # Test with insufficient funds for fixed amount
        size = self.bot._calculate_position_size(pos_sizing_fixed, 3000, 100)
        self.assertEqual(size, 30)  # Limited by available cash
    
    def test_save_and_load(self):
        """Test saving and loading bot state"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            # Save bot state
            filename = self.bot.save(tmp.name)
            self.assertEqual(filename, tmp.name)
            
            # Load bot state
            loaded_bot = Bot.load(filename)
            
            # Check loaded state
            self.assertEqual(loaded_bot.ticker, self.bot.ticker)
            self.assertEqual(loaded_bot.parameters['initial_capital'], self.bot.parameters['initial_capital'])
            self.assertEqual(loaded_bot.best_strategy.grammar_string, self.bot.best_strategy.grammar_string)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_strategy_view(self):
        """Test the strategy viewing functionality"""
        # Populate some data for the best strategy
        self.bot.best_strategy.profit_percentage = 15.0
        self.bot.best_strategy.metrics['sharpe_ratio'] = 1.5
        self.bot.best_strategy.metrics['max_drawdown'] = 10.0
        
        # View results
        results = self.bot.view(detailed=False)
        
        # Check result structure
        self.assertIn('best_strategy', results)
        self.assertIn('generations', results)
        self.assertIn('strategies_evaluated', results)
        
        # Check best strategy details
        best_strategy = results['best_strategy']
        self.assertEqual(best_strategy['grammar_string'], "BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70")
        self.assertEqual(best_strategy['profit_percentage'], 15.0)

class TestBotPositionSizing(unittest.TestCase):
    """Test cases for Bot position sizing calculation"""
    
    def test_position_size_calculation(self):
        """Test position size calculation logic"""
        # Create a Bot instance directly
        bot = Bot('AAPL', start_date='2022-01-01', end_date='2022-01-31')
        
        # Test fixed amount
        pos_sizing_fixed = {'type': 'FIXED', 'amount': 5000}
        size = bot._calculate_position_size(pos_sizing_fixed, 10000, 100)
        self.assertEqual(size, 50)  # 5000/100 = 50 shares
        
        # Test percentage
        pos_sizing_pct = {'type': 'PERCENTAGE', 'percentage': 50}
        size = bot._calculate_position_size(pos_sizing_pct, 10000, 100)
        self.assertEqual(size, 50)  # 50% of 10000 = 5000, 5000/100 = 50 shares
        
        # Test risk-based
        pos_sizing_risk = {'type': 'RISK_BASED', 'risk_percentage': 2}
        size = bot._calculate_position_size(pos_sizing_risk, 10000, 100)
        self.assertEqual(size, 2)  # 2% of 10000 = 200, 200/100 = 2 shares
        
        # Test with insufficient funds for fixed amount
        size = bot._calculate_position_size(pos_sizing_fixed, 3000, 100)
        self.assertEqual(size, 30)  # Limited by available cash
        
        # Clean up resources
        bot.delete()

if __name__ == '__main__':
    unittest.main() 