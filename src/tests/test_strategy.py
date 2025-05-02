import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy import Strategy
from grammar_parser import GrammarParser

class TestStrategy(unittest.TestCase):
    """Test cases for the Strategy class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = GrammarParser()
        self.strategy = Strategy("BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70", generation=1)
        
        # Create sample portfolio history for metric calculations
        dates = pd.date_range(start='2022-01-01', periods=100)
        values = np.linspace(10000, 12000, 100)  # Linear growth from 10k to 12k
        
        # Add some volatility
        np.random.seed(42)
        noise = np.random.normal(0, 100, 100)
        values = values + noise
        
        self.portfolio_history = pd.Series(values, index=dates)
        
        # Create sample trades
        self.sample_trades = [
            {'type': 'BUY', 'date': dates[10], 'price': 100, 'shares': 50, 'profit': 0},
            {'type': 'SELL', 'date': dates[20], 'price': 110, 'shares': 50, 'profit': 500},  # Win
            {'type': 'BUY', 'date': dates[30], 'price': 95, 'shares': 60, 'profit': 0},
            {'type': 'SELL', 'date': dates[40], 'price': 105, 'shares': 60, 'profit': 600},  # Win
            {'type': 'BUY', 'date': dates[50], 'price': 100, 'shares': 55, 'profit': 0},
            {'type': 'SELL', 'date': dates[60], 'price': 90, 'shares': 55, 'profit': -550},  # Loss
        ]
    
    def test_initialization(self):
        """Test strategy initialization"""
        strategy = Strategy("BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70", generation=5)
        
        self.assertEqual(strategy.grammar_string, "BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70")
        self.assertEqual(strategy.generation, 5)
        self.assertEqual(strategy.initial_portfolio, 0.0)
        self.assertEqual(strategy.final_portfolio, 0.0)
        self.assertEqual(strategy.profit_percentage, 0.0)
        self.assertEqual(strategy.improvement, 0.0)
        self.assertEqual(len(strategy.trades), 0)
        self.assertIsNone(strategy.parsed_strategy)
    
    def test_parsing(self):
        """Test strategy parsing"""
        self.strategy.parse(self.parser)
        
        self.assertIsNotNone(self.strategy.parsed_strategy)
        self.assertIn('entry_conditions', self.strategy.parsed_strategy)
        self.assertIn('exit_conditions', self.strategy.parsed_strategy)
    
    def test_metric_calculation(self):
        """Test performance metric calculations"""
        # Add trades to strategy
        self.strategy.trades = self.sample_trades
        
        # Calculate metrics
        metrics = self.strategy.calculate_metrics(self.portfolio_history)
        
        # Check calculated values
        self.assertEqual(self.strategy.initial_portfolio, self.portfolio_history.iloc[0])
        self.assertEqual(self.strategy.final_portfolio, self.portfolio_history.iloc[-1])
        
        # Profit percentage should be roughly 20% (from 10k to 12k)
        self.assertAlmostEqual(self.strategy.profit_percentage, 20.0, delta=1.0)
        
        # Check metrics dictionary
        self.assertGreater(metrics['sharpe_ratio'], 0)
        self.assertGreater(metrics['max_drawdown'], 0)
        
        # Win rate should be 2/3 as we have 2 winning trades and 1 losing trade
        self.assertAlmostEqual(metrics['win_rate'], 66.67, delta=0.1)
    
    def test_strategy_comparison(self):
        """Test strategy comparison logic"""
        strategy1 = Strategy("Strategy 1", generation=1)
        strategy2 = Strategy("Strategy 2", generation=1)
        
        # Set profit percentages
        strategy1.profit_percentage = 10.0
        strategy2.profit_percentage = 15.0
        
        # Set Sharpe ratios
        strategy1.metrics['sharpe_ratio'] = 1.5
        strategy2.metrics['sharpe_ratio'] = 1.2
        
        # Set max drawdowns
        strategy1.metrics['max_drawdown'] = 5.0
        strategy2.metrics['max_drawdown'] = 8.0
        
        # Strategy 2 should be better (higher profit)
        self.assertFalse(strategy1.compare_to(strategy2))
        self.assertTrue(strategy2.compare_to(strategy1))
        
        # Test for equal profit but different Sharpe
        strategy2.profit_percentage = 10.0
        self.assertTrue(strategy1.compare_to(strategy2))  # Strategy 1 has better Sharpe
        
        # Test for equal profit and Sharpe but different drawdown
        strategy2.metrics['sharpe_ratio'] = 1.5
        self.assertTrue(strategy1.compare_to(strategy2))  # Strategy 1 has lower drawdown
    
    def test_serialization(self):
        """Test strategy serialization and deserialization"""
        # Setup strategy with data
        self.strategy.initial_portfolio = 10000.0
        self.strategy.final_portfolio = 12000.0
        self.strategy.profit_percentage = 20.0
        self.strategy.improvement = 5.0
        self.strategy.trades = self.sample_trades
        self.strategy.metrics['sharpe_ratio'] = 1.5
        self.strategy.metrics['max_drawdown'] = 10.0
        
        # Convert to dictionary
        strategy_dict = self.strategy.to_dict()
        
        # Check dictionary values
        self.assertEqual(strategy_dict['grammar_string'], self.strategy.grammar_string)
        self.assertEqual(strategy_dict['profit_percentage'], 20.0)
        self.assertEqual(strategy_dict['metrics']['sharpe_ratio'], 1.5)
        
        # Convert to JSON and back
        json_str = self.strategy.to_json()
        restored_strategy = Strategy.from_json(json_str)
        
        # Check restored values
        self.assertEqual(restored_strategy.grammar_string, self.strategy.grammar_string)
        self.assertEqual(restored_strategy.profit_percentage, self.strategy.profit_percentage)
        self.assertEqual(restored_strategy.metrics['sharpe_ratio'], self.strategy.metrics['sharpe_ratio'])
        self.assertEqual(len(restored_strategy.trades), len(self.strategy.trades))

if __name__ == '__main__':
    unittest.main() 