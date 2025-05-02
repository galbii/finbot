import pandas as pd
import requests
from typing import Optional, Union
from datetime import datetime, timedelta
import time
from typing import Dict

class DataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DataFetcher class.
        
        Args:
            api_key (str, optional): Alpha Vantage API key. If not provided, will look for ALPHA_VANTAGE_API_KEY environment variable.
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.last_call_time = 0
        self.min_call_interval = 12  # Minimum seconds between API calls (Alpha Vantage limit is 5 calls per minute)
        
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            time.sleep(sleep_time)
            
        self.last_call_time = time.time()
        
    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a given symbol using Alpha Vantage API.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL' for Apple)
            start_date (str or datetime, optional): Start date in 'YYYY-MM-DD' format or datetime object
            end_date (str or datetime, optional): End date in 'YYYY-MM-DD' format or datetime object
            interval (str): Data interval ('1d' for daily, '1wk' for weekly, '1mo' for monthly)
            
        Returns:
            pd.DataFrame: DataFrame containing historical OHLCV data with standardized column names
        """
        if self.api_key is None:
            raise ValueError(
                "Alpha Vantage API key is required. Get a free key at: "
                "https://www.alphavantage.co/support/#api-key"
            )
            
        # Map intervals to Alpha Vantage function names (using free endpoints)
        interval_mapping = {
            "1d": "TIME_SERIES_DAILY",  # Changed from DAILY_ADJUSTED
            "1wk": "TIME_SERIES_WEEKLY",
            "1mo": "TIME_SERIES_MONTHLY"
        }
        
        if interval not in interval_mapping:
            raise ValueError(f"Invalid interval. Must be one of: {list(interval_mapping.keys())}")
            
        # Prepare API parameters
        params = {
            "function": interval_mapping[interval],
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        try:
            # Wait for rate limit if necessary
            self._wait_for_rate_limit()
            
            # Make API request with retries
            max_retries = 3
            retry_delay = 20  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Check for API errors
                    if "Error Message" in data:
                        raise ValueError(data["Error Message"])
                    if "Information" in data:
                        if "API call frequency" in data["Information"]:
                            # Rate limit hit, wait and retry
                            time.sleep(retry_delay)
                            continue
                        raise ValueError(f"API Error: {data['Information']}")
                        
                    break  # Success, exit retry loop
                    
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise
                    time.sleep(retry_delay)
                    continue
            
            # Get the time series data
            time_series_key = [k for k in data.keys() if "Time Series" in k][0]
            time_series_data = data[time_series_key]
            
            # Convert to DataFrame with consistent column names
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            
            # Standardize column names to match EvolvedAppleBot expectations
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume',
            }
            df = df.rename(columns=column_mapping)
            
            # Convert index to datetime and reset
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={'index': 'date'})
            
            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # Filter by date range if provided
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date]
                
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            if df.empty:
                raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")
                
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")

# Example usage
if __name__ == "__main__":
    import os
    
    # Get API key from environment variable
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if api_key is None:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
        print("Get a free API key at: https://www.alphavantage.co/support/#api-key")
        exit(1)
        
    fetcher = DataFetcher(api_key=api_key)
    
    try:
        # Test with AAPL for the last month
        symbol = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"\nTesting {symbol}:")
        print("-" * 50)
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            df = fetcher.get_historical_data(symbol, start_date=start_date, end_date=end_date)
            print(f"Successfully fetched {len(df)} days of data")
            print("\nFirst few rows:")
            print(df.head())
            print("\nLast few rows:")
            print(df.tail())
            print("\nColumns in the dataframe:")
            print(df.columns.tolist())
            print("\nData summary:")
            print(df.describe())
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            
    except Exception as e:
        print(f"General error: {str(e)}") 