import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import time
import requests
import glob
from .polygon_data_source import PolygonDataSource

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataManager')

class DataManager:
    """
    Handles downloading, processing, and storing of financial data.
    
    This class is responsible for:
    - Fetching historical OHLCV data from financial APIs
    - Validating and cleaning the data
    - Storing data for efficient retrieval
    """
    
    def __init__(self, data_dir='./data', polygon_api_key='VjtWT8rQOMD4ltn2AxVCTgxgXl21YhrP'):
        """
        Initialize the DataManager with a directory for data storage.
        
        Args:
            data_dir (str): Directory to store downloaded data
            polygon_api_key (str, optional): API key for Polygon.io
        """
        self.data_dir = data_dir
        self._ensure_data_dir()
        self.data_cache = {}  # In-memory cache for frequently accessed data
        
        # Initialize data sources
        self.polygon_api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY')
        if self.polygon_api_key:
            self.polygon_data_source = PolygonDataSource(self.polygon_api_key)
            logger.info("Initialized Polygon.io data source")
        else:
            self.polygon_data_source = None
            logger.warning("No Polygon.io API key provided, will fall back to Yahoo Finance")
        
        logger.info(f"DataManager initialized with data directory: {data_dir}")
    
    def _ensure_data_dir(self):
        """Create the data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
        
        # Also ensure the project data directory exists if it's different
        project_data_dir = "./data"
        if project_data_dir != self.data_dir and not os.path.exists(project_data_dir):
            os.makedirs(project_data_dir)
            logger.info(f"Created project data directory: {project_data_dir}")
    
    def _validate_data(self, df):
        """
        Validate the fetched data for common issues.
        
        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check for empty dataframe
        if df.empty:
            logger.error("Data validation failed: Empty dataframe")
            return False
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Data contains {df.isnull().sum().sum()} missing values")
        
        # Check for duplicate indices
        if df.index.duplicated().any():
            logger.error("Data validation failed: Contains duplicate dates")
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Data validation failed: Missing required columns. Required: {required_columns}, Found: {df.columns}")
            return False
        
        # Check for chronological order
        if not df.index.is_monotonic_increasing:
            logger.error("Data validation failed: Dates are not in chronological order")
            return False
        
        return True
    
    def _clean_data(self, df):
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Handle missing values
        if df.isnull().any().any():
            # For OHLCV data, forward filling is often appropriate
            df = df.ffill()
            # If there are still NaNs (e.g., at the beginning), use backward filling
            df = df.bfill()
            logger.info("Cleaned missing values in data")
        
        # Ensure float data type for numerical columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    def fetch_data(self, ticker, start_date, end_date, use_cache=True, max_retries=3):
        """
        Fetch historical OHLCV data for the given ticker and date range.
        
        Args:
            ticker (str): The stock symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            use_cache (bool, optional): Whether to use cached data. Defaults to True.
            max_retries (int, optional): Maximum number of retries for downloads. Defaults to 3.
            
        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex and columns [Open, High, Low, Close, Volume]
        """
        # Standardize ticker format
        ticker = ticker.upper().strip()
        
        # Check if data is in cache
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if use_cache and cache_key in self.data_cache:
            logger.info(f"Using cached data for {ticker} from {start_date} to {end_date}")
            return self.data_cache[cache_key]
        
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Try using Polygon.io first if API key is available
        if hasattr(self, 'polygon_data_source') and self.polygon_data_source is not None:
            try:
                logger.info(f"Attempting to fetch data from Polygon.io for {ticker}")
                df = self.polygon_data_source.fetch_historical_data(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched data from Polygon.io for {ticker}")
                    if self._validate_data(df):
                        if use_cache:
                            self.data_cache[cache_key] = df
                        return df
                else:
                    logger.warning(f"No data returned from Polygon.io for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to fetch data from Polygon.io: {str(e)}")
        else:
            logger.info("Polygon.io data source not available")
        
        # Try Yahoo Finance as fallback if Polygon.io fails
        try:
            logger.info(f"Attempting to fetch data from Yahoo Finance for {ticker}")
            df = self._download_from_yahoo(ticker, start_date, end_date, max_retries)
            
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched data from Yahoo Finance for {ticker}")
                if self._validate_data(df):
                    if use_cache:
                        self.data_cache[cache_key] = df
                    return df
            else:
                logger.warning(f"No data returned from Yahoo Finance for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to fetch data from Yahoo Finance: {str(e)}")
        
        # Fall back to sample data if all else fails
        logger.warning(f"All data sources failed for {ticker}, using sample data")
        df = self._get_sample_data(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            logger.info(f"Using sample/synthetic data for {ticker}")
            if use_cache:
                self.data_cache[cache_key] = df
            return df
        
        # This should never happen now with the synthetic data generation
        logger.error(f"Failed to fetch data for {ticker} from all sources")
        return None
    
    def _download_from_yahoo(self, ticker, start_date, end_date, max_retries=3):
        """
        Download data from Yahoo Finance, attempting multiple methods with retries.
        
        This method tries two approaches in the following order:
        1. Using the yfinance package (preferred method)
        2. Direct API download as fallback
        
        Args:
            ticker (str): The stock symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            max_retries (int): Maximum number of retries for each method
            
        Returns:
            pd.DataFrame: Downloaded data or None if all methods failed
        """
        # First try using yfinance library
        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(f"Attempt {retry_count+1}/{max_retries} to download {ticker} data using yfinance")
                df = self._download_from_yfinance(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    logger.info(f"Successfully downloaded {ticker} data using yfinance")
                    return self._clean_data(df)
                else:
                    logger.warning(f"yfinance returned empty dataset for {ticker}")
            except Exception as e:
                logger.warning(f"yfinance download attempt {retry_count+1} failed: {e}")
            
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        # If yfinance failed, try direct API download
        logger.info(f"yfinance download failed after {max_retries} attempts, trying direct API...")
        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(f"Attempt {retry_count+1}/{max_retries} to download {ticker} data using direct API")
                df = self._download_alternative(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    logger.info(f"Successfully downloaded {ticker} data using direct API")
                    return self._clean_data(df)
                else:
                    logger.warning(f"Direct API returned empty dataset for {ticker}")
            except Exception as e:
                logger.warning(f"Direct API download attempt {retry_count+1} failed: {e}")
            
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        logger.error(f"All Yahoo Finance download methods failed for {ticker}")
        return None
    
    def _download_from_yfinance(self, ticker, start_date, end_date):
        """
        Download data using yfinance.
        
        Args:
            ticker (str): The stock symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: Downloaded data or None if failed
        """
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            return df
        except Exception as e:
            logger.error(f"yfinance download error: {e}")
            return None
    
    def _download_alternative(self, ticker, start_date, end_date):
        """
        Download data using alternative method.
        
        Args:
            ticker (str): The stock symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: Downloaded data or None if failed
        """
        # Create direct URL for Yahoo Finance
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
        
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_timestamp}&period2={end_timestamp}&interval=1d&events=history"
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Create DataFrame from CSV content
            df = pd.read_csv(pd.io.common.StringIO(response.text), index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"Alternative download error: {e}")
            return None
    
    def _get_sample_data(self, ticker, start_date=None, end_date=None):
        """
        Get sample data for the ticker.
        This is mainly used for testing or when real data fetching fails.
        
        Args:
            ticker (str): The stock symbol
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame containing sample data
        """
        # Define path to sample data file
        sample_dir = os.path.dirname(os.path.abspath(__file__))
        sample_file = os.path.join(sample_dir, f"sample_{ticker}.csv")
        
        # If sample file for this ticker doesn't exist, use AAPL as fallback
        if not os.path.exists(sample_file):
            logger.warning(f"No sample data for {ticker}, using AAPL sample data instead")
            sample_file = os.path.join(sample_dir, "sample_AAPL.csv")
        
        # If still no sample file, create synthetic data
        if not os.path.exists(sample_file):
            logger.warning("No sample data files found, generating synthetic data")
            return self._generate_synthetic_data(start_date, end_date)
        
        try:
            # Load sample data
            df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
            
            # Validate and filter date range if necessary
            if self._validate_data(df):
                if start_date and end_date:
                    # Filter to requested date range
                    df = df.loc[start_date:end_date]
                    
                    # If no data in range, generate synthetic data
                    if df.empty:
                        logger.warning(f"No sample data for date range {start_date} to {end_date}, generating synthetic data")
                        return self._generate_synthetic_data(start_date, end_date)
                
                return df
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
        
        # If all else fails, generate synthetic data
        return self._generate_synthetic_data(start_date, end_date)
    
    def _generate_synthetic_data(self, start_date=None, end_date=None):
        """
        Generate synthetic OHLCV data for testing purposes.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with synthetic data
        """
        logger.info("Generating synthetic price data for testing")
        
        # Default to last year of data if no dates provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate random price data with realistic behavior
        np.random.seed(42)  # For reproducibility
        
        # Start with a price around 100
        price = 100
        prices = [price]
        
        # Generate daily returns with realistic distribution
        for _ in range(1, len(date_range)):
            daily_return = np.random.normal(0.0005, 0.015)  # Mean positive drift, realistic volatility
            price = price * (1 + daily_return)
            prices.append(price)
        
        prices = np.array(prices)
        
        # Generate OHLC based on close prices
        df = pd.DataFrame({
            'Close': prices,
            'Open': prices * (1 + np.random.normal(0, 0.005, len(prices))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
            'Volume': np.random.randint(1000000, 10000000, len(prices))
        }, index=date_range)
        
        # Make sure High is highest and Low is lowest for each day
        for i in range(len(df)):
            row = df.iloc[i]
            high = max(row['Open'], row['Close'], row['High'])
            low = min(row['Open'], row['Close'], row['Low'])
            df.loc[df.index[i], 'High'] = high
            df.loc[df.index[i], 'Low'] = low
        
        logger.info(f"Generated synthetic data with {len(df)} data points")
        return df
    
    def get_data(self, ticker, start_date=None, end_date=None):
        """
        Get data for a ticker, using cached data if available.
        
        Args:
            ticker (str): The stock symbol
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame containing the historical data
        """
        return self.fetch_data(ticker, start_date, end_date, use_cache=True)
    
    def get_versioned_dataset(self, ticker):
        """
        Get the latest versioned dataset for a ticker.
        
        Args:
            ticker (str): The stock symbol
            
        Returns:
            pd.DataFrame: DataFrame containing the latest historical data
        """
        ticker = ticker.upper().strip()
        latest_file = self.get_latest_dataset_file(ticker)
        
        if latest_file:
            logger.info(f"Loading existing dataset: {latest_file}")
            try:
                df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
                if self._validate_data(df):
                    return df
                else:
                    logger.warning(f"Invalid data in {latest_file}, will download fresh data")
            except Exception as e:
                logger.error(f"Error loading dataset {latest_file}: {e}")
        
        # If we don't have a valid dataset, download a fresh one
        return self.download_latest_dataset(ticker)
    
    def get_latest_dataset_file(self, ticker):
        """
        Find the latest dataset file for a ticker.
        
        Args:
            ticker (str): The stock symbol
            
        Returns:
            str: Path to the latest dataset file, or None if not found
        """
        ticker = ticker.upper().strip()
        pattern = os.path.join(self.data_dir, f"{ticker}*.csv")
        files = glob.glob(pattern)
        
        if not files:
            logger.info(f"No existing dataset found for {ticker}")
            return None
        
        # Sort files by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        latest_file = files[0]
        
        # Check if the latest file is recent enough (within 1 day)
        mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
        now = datetime.now()
        
        if (now - mod_time).days > 1:
            logger.info(f"Latest dataset for {ticker} is older than 1 day: {mod_time.strftime('%Y-%m-%d')}")
            return None
        
        return latest_file
    
    def download_latest_dataset(self, ticker):
        """
        Download the latest dataset for a ticker and save it with date version.
        
        Args:
            ticker (str): The stock symbol
            
        Returns:
            pd.DataFrame: DataFrame containing the latest historical data
        """
        ticker = ticker.upper().strip()
        
        # Get current date for end_date and go back 2 years for start_date
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        logger.info(f"Downloading latest dataset for {ticker} from {start_date} to {end_date}")
        
        # Fetch the data
        df = self.fetch_data(ticker, start_date, end_date, use_cache=False)
        
        if df is None or df.empty:
            logger.error(f"Failed to download data for {ticker}")
            return None
        
        # Get the date of the last entry to use in the filename
        last_date = df.index[-1].strftime('%m%d')
        filename = os.path.join(self.data_dir, f"{ticker}{last_date}.csv")
        
        # Save the dataset
        try:
            df.to_csv(filename)
            logger.info(f"Saved dataset to {filename}")
            
            # Clean up old dataset files for this ticker (keep max 3)
            self._cleanup_old_datasets(ticker)
            
            return df
        except Exception as e:
            logger.error(f"Error saving dataset to {filename}: {e}")
            return df
    
    def is_dataset_up_to_date(self, ticker):
        """
        Check if the dataset for a ticker is up-to-date.
        
        Args:
            ticker (str): The stock symbol
            
        Returns:
            bool: True if dataset is up-to-date, False otherwise
        """
        ticker = ticker.upper().strip()
        latest_file = self.get_latest_dataset_file(ticker)
        
        if not latest_file:
            return False
        
        # Check the date of the last entry in the dataset
        try:
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            last_date = df.index[-1]
            current_date = datetime.now().date()
            
            # If the dataset's last date is more than 1 day old (accounting for weekends/holidays)
            # and it's not a weekend, consider it outdated
            if (current_date - last_date.date()).days > 3:
                logger.info(f"Dataset for {ticker} is outdated: last entry is {last_date.date()}")
                return False
                
            # Check if the dataset is from a previous trading day
            if current_date.weekday() >= 1 and last_date.date() < current_date - timedelta(days=1):
                # This is a weekday and the data is not from yesterday
                logger.info(f"Dataset for {ticker} is outdated: last entry is {last_date.date()}")
                return False
                
            # If today is Monday, check if data is from Friday or later
            if current_date.weekday() == 0 and last_date.date() < current_date - timedelta(days=3):
                logger.info(f"Dataset for {ticker} is outdated: last entry is {last_date.date()}")
                return False
                
            logger.info(f"Dataset for {ticker} is up-to-date: last entry is {last_date.date()}")
            return True
        except Exception as e:
            logger.error(f"Error checking dataset date for {ticker}: {e}")
            return False
    
    def _cleanup_old_datasets(self, ticker, keep=3):
        """
        Clean up old dataset files for a ticker, keeping the most recent ones.
        
        Args:
            ticker (str): The stock symbol
            keep (int): Number of recent files to keep
        """
        ticker = ticker.upper().strip()
        pattern = os.path.join(self.data_dir, f"{ticker}*.csv")
        files = glob.glob(pattern)
        
        if len(files) <= keep:
            return
        
        # Sort files by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Delete older files
        for file in files[keep:]:
            try:
                os.remove(file)
                logger.info(f"Deleted old dataset file: {file}")
            except Exception as e:
                logger.error(f"Error deleting file {file}: {e}")
    
    def clear_cache(self):
        """Clear the in-memory data cache"""
        self.data_cache.clear()
        logger.info("Data cache cleared") 