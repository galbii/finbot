import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.indicators.indicator_calculator import IndicatorCalculator

class TestIndicatorCalculator(unittest.TestCase):
    """Test cases for the IndicatorCalculator class"""
    
    def setUp(self):
        """Set up test environment"""
        self.indicator_calculator = IndicatorCalculator()
        
        # Create a sample OHLCV dataframe for testing
        date_rng = pd.date_range(start='2020-01-01', end='2020-03-01', freq='D')
        self.test_df = pd.DataFrame({
            'Open': np.random.rand(len(date_rng)) * 100 + 100,
            'High': np.random.rand(len(date_rng)) * 100 + 110,
            'Low': np.random.rand(len(date_rng)) * 100 + 90,
            'Close': np.random.rand(len(date_rng)) * 100 + 100,
            'Volume': np.random.randint(1000, 10000, size=len(date_rng))
        }, index=date_rng)
    
    def tearDown(self):
        """Clean up test environment"""
        self.indicator_calculator.clear_cache()
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.indicator_calculator.calculate_rsi(self.test_df, period=14)
        
        # Check that result is a pandas Series
        self.assertIsInstance(rsi, pd.Series)
        
        # Check length (should be same as input with some initial NaNs)
        self.assertEqual(len(rsi), len(self.test_df))
        
        # Check range (RSI should be between 0 and 100)
        self.assertTrue(all(0 <= val <= 100 for val in rsi.dropna()))
        
        # Check that initial values are NaN (should be period - 1 for ta library implementation)
        self.assertEqual(rsi.isna().sum(), 13)  # RSI from ta library has period-1 NaN values
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd = self.indicator_calculator.calculate_macd(
            self.test_df, 
            fast_period=12, 
            slow_period=26, 
            signal_period=9
        )
        
        # Check that result is a pandas DataFrame
        self.assertIsInstance(macd, pd.DataFrame)
        
        # Check that it has the right columns
        self.assertIn('macd', macd.columns)
        self.assertIn('signal', macd.columns)
        self.assertIn('histogram', macd.columns)
        
        # Check length (should be same as input with some initial NaNs)
        self.assertEqual(len(macd), len(self.test_df))
        
        # Check that initial values are NaN
        self.assertTrue(macd['macd'].isna().sum() > 0)
        self.assertTrue(macd['signal'].isna().sum() > 0)
        self.assertTrue(macd['histogram'].isna().sum() > 0)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        bb = self.indicator_calculator.calculate_bollinger_bands(
            self.test_df, 
            window=20, 
            std=2
        )
        
        # Check that result is a pandas DataFrame
        self.assertIsInstance(bb, pd.DataFrame)
        
        # Check that it has the right columns
        self.assertIn('upper', bb.columns)
        self.assertIn('middle', bb.columns)
        self.assertIn('lower', bb.columns)
        
        # Check length (should be same as input with some initial NaNs)
        self.assertEqual(len(bb), len(self.test_df))
        
        # Check that initial values are NaN
        self.assertTrue(bb['upper'].isna().sum() > 0)
        
        # Check relationships between bands (upper > middle > lower)
        valid_indices = ~bb['upper'].isna()
        self.assertTrue(all(bb.loc[valid_indices, 'upper'] >= bb.loc[valid_indices, 'middle']))
        self.assertTrue(all(bb.loc[valid_indices, 'middle'] >= bb.loc[valid_indices, 'lower']))
    
    def test_sma_calculation(self):
        """Test SMA calculation"""
        period = 20
        sma = self.indicator_calculator.calculate_sma(self.test_df, period=period)
        
        # Check that result is a pandas Series
        self.assertIsInstance(sma, pd.Series)
        
        # Check length (should be same as input with some initial NaNs)
        self.assertEqual(len(sma), len(self.test_df))
        
        # Check that initial values are NaN (should match period)
        self.assertEqual(sma.isna().sum(), period - 1)
        
        # Manual calculation for verification
        manual_sma = self.test_df['Close'].rolling(window=period).mean()
        pd.testing.assert_series_equal(sma, manual_sma, check_names=False)
    
    def test_ema_calculation(self):
        """Test EMA calculation"""
        period = 20
        ema = self.indicator_calculator.calculate_ema(self.test_df, period=period)
        
        # Check that result is a pandas Series
        self.assertIsInstance(ema, pd.Series)
        
        # Check length (should be same as input with some initial NaNs)
        self.assertEqual(len(ema), len(self.test_df))
        
        # Check that initial values are NaN
        self.assertTrue(ema.isna().sum() > 0)
    
    def test_crossover_detection(self):
        """Test crossover detection"""
        # Create two series with a single known crossover point
        date_rng = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        # Design series with exactly one crossover at index 3
        series1 = pd.Series([5, 6, 7, 12, 13, 14, 15, 16, 17, 18], index=date_rng)
        series2 = pd.Series([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], index=date_rng)
        
        # Calculate crossover
        crossover = self.indicator_calculator.detect_crossover(series1, series2)
        
        # Check specific point - should be True at index 3 (2020-01-04)
        self.assertTrue(crossover.iloc[3])
        
        # Count total crossovers - should be exactly 1
        self.assertEqual(crossover.sum(), 1)
    
    def test_crossunder_detection(self):
        """Test crossunder detection"""
        # Create two series with a single known crossunder point
        date_rng = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        # Design series with exactly one crossunder at index 3
        series1 = pd.Series([15, 14, 13, 8, 7, 6, 5, 4, 3, 2], index=date_rng)
        series2 = pd.Series([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], index=date_rng)
        
        # Calculate crossunder
        crossunder = self.indicator_calculator.detect_crossunder(series1, series2)
        
        # Check specific point - should be True at index 3 (2020-01-04)
        self.assertTrue(crossunder.iloc[3])
        
        # Count total crossunders - should be exactly 1
        self.assertEqual(crossunder.sum(), 1)
    
    def test_indicator_caching(self):
        """Test that indicators are properly cached"""
        # First calculation should compute
        rsi1 = self.indicator_calculator.calculate_indicator(self.test_df, 'RSI', period=14)
        
        # Get the cache size
        initial_cache_size = len(self.indicator_calculator.indicator_cache)
        self.assertEqual(initial_cache_size, 1)
        
        # Second calculation should use cache
        rsi2 = self.indicator_calculator.calculate_indicator(self.test_df, 'RSI', period=14)
        
        # Cache size should remain the same
        self.assertEqual(len(self.indicator_calculator.indicator_cache), initial_cache_size)
        
        # Both RSI results should be identical
        pd.testing.assert_series_equal(rsi1, rsi2)
        
        # Different parameters should create new cache entry
        rsi3 = self.indicator_calculator.calculate_indicator(self.test_df, 'RSI', period=7)
        self.assertEqual(len(self.indicator_calculator.indicator_cache), initial_cache_size + 1)
        
        # Clear cache
        self.indicator_calculator.clear_cache()
        self.assertEqual(len(self.indicator_calculator.indicator_cache), 0)

if __name__ == '__main__':
    unittest.main() 