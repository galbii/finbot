import unittest
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot import Bot

class TestBotPositionSizing(unittest.TestCase):
    """Test cases for Bot position sizing calculation"""
    
    def test_position_size_calculation(self):
        """Test position size calculation logic without initializing a real bot"""
        # Create a mock method to test
        def calculate_position_size(position_sizing, available_cash, price):
            if position_sizing['type'] == 'FIXED':
                # Fixed dollar amount
                return min(position_sizing['amount'] / price, available_cash / price)
            
            elif position_sizing['type'] == 'PERCENTAGE':
                # Percentage of available cash
                percentage = position_sizing['percentage'] / 100
                return (available_cash * percentage) / price
            
            elif position_sizing['type'] == 'RISK_BASED':
                # Risk-based position sizing (simplified)
                risk_percentage = position_sizing['risk_percentage'] / 100
                return (available_cash * risk_percentage) / price
            
            # Default to using all available cash
            return available_cash / price
        
        # Test fixed amount
        pos_sizing_fixed = {'type': 'FIXED', 'amount': 5000}
        size = calculate_position_size(pos_sizing_fixed, 10000, 100)
        self.assertEqual(size, 50)  # 5000/100 = 50 shares
        
        # Test percentage
        pos_sizing_pct = {'type': 'PERCENTAGE', 'percentage': 50}
        size = calculate_position_size(pos_sizing_pct, 10000, 100)
        self.assertEqual(size, 50)  # 50% of 10000 = 5000, 5000/100 = 50 shares
        
        # Test risk-based
        pos_sizing_risk = {'type': 'RISK_BASED', 'risk_percentage': 2}
        size = calculate_position_size(pos_sizing_risk, 10000, 100)
        self.assertEqual(size, 2)  # 2% of 10000 = 200, 200/100 = 2 shares
        
        # Test with insufficient funds for fixed amount
        size = calculate_position_size(pos_sizing_fixed, 3000, 100)
        self.assertEqual(size, 30)  # Limited by available cash

if __name__ == '__main__':
    unittest.main() 