import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting import BacktestEngine
from indicators.indicator_calculator import IndicatorCalculator
from strategy import Strategy
from grammar_parser import GrammarParser

class TestBacktesting(unittest.TestCase):
    """Test cases for the backtesting engine."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create sample historical data
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'Open': np.random.normal(100, 2, 100),
            'High': np.random.normal(102, 2, 100),
            'Low': np.random.normal(98, 2, 100),
            'Close': np.random.normal(100, 2, 100),
            'Volume': np.random.normal(1000000, 200000, 100)
        }, index=self.dates)
        
        # Ensure High > Open > Close > Low for realistic data
        for i in range(len(self.data)):
            high = max(self.data.iloc[i]['Open'], self.data.iloc[i]['Close']) + abs(np.random.normal(1, 0.5))
            low = min(self.data.iloc[i]['Open'], self.data.iloc[i]['Close']) - abs(np.random.normal(1, 0.5))
            self.data.loc[self.dates[i], 'High'] = high
            self.data.loc[self.dates[i], 'Low'] = low
        
        # Create indicator calculator
        self.indicator_calculator = IndicatorCalculator()
        
        # Create parser
        self.parser = GrammarParser()
        
        # Create a simple strategy
        self.strategy = Strategy("BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70")
        self.strategy.parse(self.parser)
        
        # Create backtesting engine
        self.engine = BacktestEngine(
            self.data, 
            self.indicator_calculator,
            params={'initial_capital': 10000}
        )
        
        # Create a mock evaluator method for the engine
        def mock_evaluate_condition(condition, data, idx):
            return False
            
        # Override the default evaluate condition method
        self.engine._evaluate_condition = mock_evaluate_condition
    
    def test_signal_generation(self):
        """Test that signals are generated correctly."""
        # Mock condition evaluator that alternates signals
        def mock_evaluator(condition, data, idx):
            # Entry signal every 10 days
            if isinstance(condition, dict) and condition.get('indicator') == 'RSI' and idx % 10 == 0:
                return True
            # Exit signal 5 days after entry
            elif isinstance(condition, dict) and condition.get('indicator') == 'RSI' and (idx - 5) % 10 == 0:
                return True
            return False
        
        # Generate signals using mock evaluator
        signals = self.engine._generate_signals(self.strategy, condition_evaluator=mock_evaluator)
        
        # Verify signals
        self.assertEqual(len(signals), len(self.data))
        # Should have at least some signals
        self.assertGreater(sum(signals['entry']), 0)
        self.assertGreater(sum(signals['exit']), 0)
    
    def test_portfolio_simulation(self):
        """Test that portfolio simulation works correctly."""
        # Create simple signals DataFrame
        signals = pd.DataFrame(index=self.data.index, columns=['entry', 'exit'])
        signals['entry'] = False
        signals['exit'] = False
        
        # Set up a simple entry and exit
        signals.iloc[40, signals.columns.get_loc('entry')] = True  # Buy on day 40 (beyond minimum data requirement)
        signals.iloc[60, signals.columns.get_loc('exit')] = True   # Sell on day 60
        
        # Initialize portfolio
        portfolio = self.engine._initialize_portfolio()
        
        # Run simulation
        result_portfolio = self.engine._simulate_portfolio(signals, portfolio)
        
        # Check portfolio has trades
        self.assertEqual(len(result_portfolio['trades']), 2)
        self.assertEqual(result_portfolio['trades'][0]['type'], 'BUY')
        self.assertEqual(result_portfolio['trades'][1]['type'], 'SELL')
        
        # Check portfolio history is recorded
        self.assertEqual(len(result_portfolio['history']), len(self.data))
    
    def test_metrics_calculation(self):
        """Test calculation of performance metrics."""
        # Create a portfolio with history
        portfolio = {
            'cash': 5000,
            'position': 0,
            'trades': [
                {'type': 'BUY', 'date': self.dates[40], 'price': 100, 'shares': 50, 'value': 5000, 'fee': 0},
                {'type': 'SELL', 'date': self.dates[60], 'price': 110, 'shares': 50, 'value': 5500, 'fee': 0, 'profit': 500, 'profit_percentage': 10}
            ],
            'history': []
        }
        
        # Generate increasing portfolio values
        for i, date in enumerate(self.dates):
            value = 10000 + i * 100  # Linearly increasing portfolio value
            portfolio['history'].append({
                'date': date,
                'cash': portfolio['cash'],
                'position': 50 if 40 <= i <= 60 else 0,
                'portfolio_value': value
            })
        
        # Calculate metrics
        metrics = self.engine._calculate_metrics(portfolio)
        
        # Check metrics
        self.assertGreater(metrics['profit_percentage'], 0)
        self.assertGreater(metrics['sharpe_ratio'], 0)
        self.assertEqual(metrics['num_trades'], 2)
        self.assertEqual(metrics['win_rate'], 100)  # 1 winning trade out of 1 sell trade
    
    def test_full_backtest(self):
        """Test the full backtesting process."""
        # Create a mock condition evaluator for testing
        def mock_evaluator(condition, data, idx):
            # Simple condition that buys at index 40 and sells at index 60
            if isinstance(condition, dict) and 'indicator' in condition:
                if condition['indicator'] == 'RSI' and idx == 40:
                    return True  # Buy signal at index 40
                elif condition['indicator'] == 'RSI' and idx == 60:
                    return True  # Sell signal at index 60
            return False
        
        # Replace the strategy with more explicit conditions
        self.strategy.parsed_strategy = {
            'entry_conditions': {'indicator': 'RSI', 'parameters': {'period': 14}, 'operator': '<', 'value': 30},
            'exit_conditions': {'indicator': 'RSI', 'parameters': {'period': 14}, 'operator': '>', 'value': 70},
            'position_sizing': None
        }
        
        # Run backtest
        results = self.engine.run_backtest(self.strategy, condition_evaluator=mock_evaluator)
        
        # Check results
        self.assertIsNotNone(results)
        self.assertIn('portfolio_history', results)
        self.assertIn('trades', results)
        self.assertIn('metrics', results)
        
        # Should have 1 buy and 1 sell trade
        buy_trades = [t for t in results['trades'] if t['type'] == 'BUY']
        sell_trades = [t for t in results['trades'] if t['type'] == 'SELL']
        
        self.assertEqual(len(buy_trades), 1)
        self.assertEqual(len(sell_trades), 1)

if __name__ == '__main__':
    unittest.main() 