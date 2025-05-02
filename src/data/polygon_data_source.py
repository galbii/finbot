import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

logger = logging.getLogger('PolygonDataSource')

class PolygonDataSource:
    """
    Data source implementation using the Polygon.io API.
    
    This class is responsible for:
    - Fetching historical OHLCV data from Polygon.io
    - Converting API responses to pandas DataFrames
    - Handling rate limiting and API errors
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key):
        """
        Initialize the Polygon.io data source.
        
        Args:
            api_key (str): Polygon.io API key
        """
        self.api_key = api_key
        logger.info("Polygon.io data source initialized")
    
    def fetch_historical_data(self, ticker, start_date=None, end_date=None, timespan="day"):
        """
        Fetch historical OHLCV data for a ticker.
        
        Args:
            ticker (str): Stock symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            timespan (str): Time interval for data (day, hour, minute)
            
        Returns:
            pd.DataFrame: DataFrame with historical data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Default to 1 year of data
            start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {timespan} data for {ticker} from {start_date} to {end_date}")
        
        # Format dates for API
        start_date_formatted = start_date.replace("-", "")
        end_date_formatted = end_date.replace("-", "")
        
        # Construct URL
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start_date_formatted}/{end_date_formatted}?apiKey={self.api_key}&sort=asc"
        
        try:
            # Make API request
            response = requests.get(url)
            
            # Check rate limiting
            if response.status_code == 429:
                logger.warning("Rate limit hit, waiting and retrying...")
                time.sleep(60)  # Wait a minute before retrying
                response = requests.get(url)
            
            # Check for successful response
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
            
            # Parse response JSON
            data = response.json()
            
            # Check if results exist
            if 'results' not in data or not data['results']:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            # Rename columns to match expected format
            column_mapping = {
                'v': 'Volume',
                'o': 'Open',
                'c': 'Close',
                'h': 'High',
                'l': 'Low',
                't': 'timestamp'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Select only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.info(f"Successfully fetched {len(df)} data points for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Polygon.io: {e}")
            return None

    def fetch_multiple_tickers(self, tickers, start_date=None, end_date=None):
        """
        Fetch historical data for multiple tickers.
        
        Args:
            tickers (list): List of stock symbols
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            dict: Dictionary mapping tickers to DataFrames
        """
        result = {}
        for ticker in tickers:
            df = self.fetch_historical_data(ticker, start_date, end_date)
            if df is not None:
                result[ticker] = df
            # Add a small delay to avoid rate limiting
            time.sleep(0.2)
        return result 