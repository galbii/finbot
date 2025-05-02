import unittest
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grammar_parser import GrammarParser

class TestGrammarParser(unittest.TestCase):
    """Test cases for the GrammarParser class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.parser = GrammarParser()
    
    def test_basic_rsi_strategy(self):
        """Test parsing a basic RSI strategy"""
        grammar_string = "BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70"
        parsed = self.parser.parse(grammar_string)
        
        # Check structure
        self.assertIn('entry_conditions', parsed)
        self.assertIn('exit_conditions', parsed)
        
        # Check entry condition
        entry = parsed['entry_conditions']
        self.assertEqual(entry['indicator'], 'RSI')
        self.assertEqual(entry['parameters']['period'], 14)
        self.assertEqual(entry['operator'], '<')
        self.assertEqual(entry['value'], 30)
        
        # Check exit condition
        exit = parsed['exit_conditions']
        self.assertEqual(exit['indicator'], 'RSI')
        self.assertEqual(exit['parameters']['period'], 14)
        self.assertEqual(exit['operator'], '>')
        self.assertEqual(exit['value'], 70)
    
    def test_position_sizing(self):
        """Test parsing position sizing rules"""
        grammar_string = "BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70 SIZE = 50%"
        parsed = self.parser.parse(grammar_string)
        
        # Check position sizing
        self.assertIn('position_sizing', parsed)
        self.assertEqual(parsed['position_sizing']['type'], 'PERCENTAGE')
        self.assertEqual(parsed['position_sizing']['percentage'], 50.0)
    
    def test_compound_conditions(self):
        """Test parsing compound conditions with AND/OR"""
        grammar_string = "BUY WHEN RSI(14) < 30 AND RSI(7) < 20 SELL WHEN RSI(14) > 70 OR RSI(7) > 80"
        parsed = self.parser.parse(grammar_string)
        
        # Check entry condition (AND)
        entry = parsed['entry_conditions']
        self.assertEqual(entry['operator'], 'AND')
        self.assertEqual(len(entry['conditions']), 2)
        
        # First subcondition
        sub1 = entry['conditions'][0]
        self.assertEqual(sub1['indicator'], 'RSI')
        self.assertEqual(sub1['parameters']['period'], 14)
        self.assertEqual(sub1['value'], 30)
        
        # Second subcondition
        sub2 = entry['conditions'][1]
        self.assertEqual(sub2['indicator'], 'RSI')
        self.assertEqual(sub2['parameters']['period'], 7)
        self.assertEqual(sub2['value'], 20)
        
        # Check exit condition (OR)
        exit = parsed['exit_conditions']
        self.assertEqual(exit['operator'], 'OR')
        self.assertEqual(len(exit['conditions']), 2)
    
    def test_macd_condition(self):
        """Test parsing MACD conditions"""
        grammar_string = "BUY WHEN MACD(12, 26, 9) CROSSES ABOVE SIGNAL SELL WHEN MACD(12, 26, 9) CROSSES BELOW 0"
        parsed = self.parser.parse(grammar_string)
        
        # Check entry condition
        entry = parsed['entry_conditions']
        self.assertEqual(entry['indicator'], 'MACD')
        self.assertEqual(entry['parameters']['fast_period'], 12)
        self.assertEqual(entry['parameters']['slow_period'], 26)
        self.assertEqual(entry['parameters']['signal_period'], 9)
        self.assertEqual(entry['operator'], 'CROSSES ABOVE')
        self.assertEqual(entry['reference'], 'SIGNAL')
        
        # Check exit condition
        exit = parsed['exit_conditions']
        self.assertEqual(exit['indicator'], 'MACD')
        self.assertEqual(exit['operator'], 'CROSSES BELOW')
        self.assertEqual(exit['reference'], '0')
    
    def test_bollinger_bands_condition(self):
        """Test parsing Bollinger Bands conditions"""
        grammar_string = "BUY WHEN PRICE < LOWER_BB(20, 2.0) SELL WHEN PRICE > UPPER_BB(20, 2.0)"
        parsed = self.parser.parse(grammar_string)
        
        # Check entry condition
        entry = parsed['entry_conditions']
        self.assertEqual(entry['indicator'], 'BOLLINGER_BANDS')
        self.assertEqual(entry['parameters']['window'], 20)
        self.assertEqual(entry['parameters']['std'], 2.0)
        self.assertEqual(entry['operator'], '<')
        self.assertEqual(entry['band'], 'LOWER')
        
        # Check exit condition
        exit = parsed['exit_conditions']
        self.assertEqual(exit['indicator'], 'BOLLINGER_BANDS')
        self.assertEqual(exit['operator'], '>')
        self.assertEqual(exit['band'], 'UPPER')
    
    def test_validation(self):
        """Test grammar validation"""
        # Valid grammar
        valid_grammar = "BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70"
        self.assertTrue(self.parser.validate(valid_grammar))
        
        # Invalid grammar - missing SELL part
        invalid_grammar1 = "BUY WHEN RSI(14) < 30"
        self.assertFalse(self.parser.validate(invalid_grammar1))
        
        # Invalid grammar - wrong syntax
        invalid_grammar2 = "BUYING RSI(14) < 30 SELLING RSI(14) > 70"
        self.assertFalse(self.parser.validate(invalid_grammar2))

if __name__ == '__main__':
    unittest.main() 