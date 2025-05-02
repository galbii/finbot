import unittest
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
import tempfile
from src.data.data_manager import DataManager

class MockDataManager(DataManager):
    """Mock version of DataManager for testing"""
    
    def fetch_data(self, ticker, start_date=None, end_date=None, force_download=False):
        """Override fetch_data to use mock data"""
        # Check if data is in cache
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self.data_cache and not force_download:
            return self.data_cache[cache_key]
        
        # Check if data is already downloaded
        file_path = os.path.join(self.data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        if os.path.exists(file_path) and not force_download:
            # Make sure to parse dates and set index correctly
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            if self._validate_data(df):
                self.data_cache[cache_key] = df
                return df
        
        # Generate mock data
        date_rng = pd.date_range(start=start_date or '2020-01-01', 
                                 end=end_date or '2020-01-31', 
                                 freq='D')
        
        # Use fixed seed for reproducibility in tests
        np.random.seed(42)
        
        mock_data = pd.DataFrame({
            'Date': date_rng,
            'Open': np.random.rand(len(date_rng)) * 100 + 100,
            'High': np.random.rand(len(date_rng)) * 100 + 110,
            'Low': np.random.rand(len(date_rng)) * 100 + 90,
            'Close': np.random.rand(len(date_rng)) * 100 + 100,
            'Volume': np.random.randint(1000, 10000, size=len(date_rng))
        })
        
        # Set the index to Date
        mock_data.set_index('Date', inplace=True)
        
        # Save to file
        file_path = os.path.join(self.data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        mock_data.to_csv(file_path)
        
        # Cache in memory
        self.data_cache[cache_key] = mock_data
        
        return mock_data

class TestDataManager(unittest.TestCase):
    """Test cases for the DataManager class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test data
        self.test_data_dir = tempfile.mkdtemp()
        self.data_manager = MockDataManager(data_dir=self.test_data_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up created test directory
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    def test_directory_creation(self):
        """Test that the data directory is created"""
        self.assertTrue(os.path.exists(self.test_data_dir))
    
    def test_fetch_data(self):
        """Test fetching data"""
        # Use a well-known ticker for testing
        ticker = 'AAPL'
        # Use a short date range to minimize test duration
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = self.data_manager.fetch_data(
            ticker, 
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify the data was fetched
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        
        # Verify required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_data_validation(self):
        """Test data validation functionality"""
        # Create a valid dataframe
        date_rng = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        valid_df = pd.DataFrame({
            'Open': np.random.rand(len(date_rng)) * 100 + 100,
            'High': np.random.rand(len(date_rng)) * 100 + 110,
            'Low': np.random.rand(len(date_rng)) * 100 + 90,
            'Close': np.random.rand(len(date_rng)) * 100 + 100,
            'Volume': np.random.randint(1000, 10000, size=len(date_rng))
        }, index=date_rng)
        
        # Test valid dataframe
        self.assertTrue(self.data_manager._validate_data(valid_df))
        
        # Test empty dataframe
        empty_df = pd.DataFrame()
        self.assertFalse(self.data_manager._validate_data(empty_df))
        
        # Test dataframe with missing required columns
        missing_col_df = valid_df.drop(columns=['Volume'])
        self.assertFalse(self.data_manager._validate_data(missing_col_df))
        
        # Test dataframe with duplicate indices
        duplicate_idx_df = pd.concat([valid_df, valid_df])
        self.assertFalse(self.data_manager._validate_data(duplicate_idx_df))
    
    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        # Create a dataframe with NaN values
        date_rng = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
        df_with_nans = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [110, 111, np.nan, 113, 114, 115, 116, 117, 118, 119],
            'Low': [90, 91, 92, np.nan, 94, 95, 96, 97, 98, 99],
            'Close': [100, 101, 102, 103, np.nan, 105, 106, 107, 108, 109],
            'Volume': [1000, 1001, 1002, 1003, 1004, np.nan, 1006, 1007, 1008, 1009]
        }, index=date_rng)
        
        # Clean the data
        cleaned_df = self.data_manager._clean_data(df_with_nans)
        
        # Verify there are no NaN values
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)
    
    def test_caching(self):
        """Test data caching functionality"""
        # Use a well-known ticker for testing
        ticker = 'AAPL'
        # Use a short date range to minimize test duration
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # First fetch should download data
        df1 = self.data_manager.fetch_data(ticker, start_date=start_date, end_date=end_date)
        
        # Second fetch should use cache
        df2 = self.data_manager.fetch_data(ticker, start_date=start_date, end_date=end_date)
        
        # Both should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Check cache directly
        cache_key = f"{ticker}_{start_date}_{end_date}"
        self.assertIn(cache_key, self.data_manager.data_cache)
        
        # Clear cache and verify it's empty
        self.data_manager.clear_cache()
        self.assertEqual(len(self.data_manager.data_cache), 0)
    
    def test_file_storage(self):
        """Test that data is stored to and loaded from files"""
        # Use a well-known ticker for testing
        ticker = 'AAPL'
        # Use fixed dates for reproducibility
        start_date = '2020-01-01'
        end_date = '2020-01-31'
        
        # Fetch data (should generate and save to file)
        df1 = self.data_manager.fetch_data(ticker, start_date=start_date, end_date=end_date)
        
        # Clear cache
        self.data_manager.clear_cache()
        
        # Fetch again (should load from file, not generate new data)
        df2 = self.data_manager.fetch_data(ticker, start_date=start_date, end_date=end_date)
        
        # Both should be identical
        pd.testing.assert_frame_equal(df1, df2)
        
        # Verify file exists
        file_path = os.path.join(self.test_data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        self.assertTrue(os.path.exists(file_path))

if __name__ == '__main__':
    unittest.main() 