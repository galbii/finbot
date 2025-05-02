#!/usr/bin/env python3
"""
Script to test fetching data from Polygon.io API.
This is useful for testing the API key and connection.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PolygonTest')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.polygon_data_source import PolygonDataSource

def main():
    parser = argparse.ArgumentParser(description='Test Polygon.io data fetching')
    parser.add_argument('--api-key', type=str, help='Polygon.io API key')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol to fetch')
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to fetch')
    parser.add_argument('--timespan', type=str, default='day', help='Timespan (day, hour, minute)')
    parser.add_argument('--historical', action='store_true', help='Use historical date range (2 years ago)')
    args = parser.parse_args()
    
    # Get API key from args, env var, or default
    api_key = args.api_key or os.environ.get('POLYGON_API_KEY', 'VjtWT8rQOMD4ltn2AxVCTgxgXl21YhrP')
    
    if not api_key:
        logger.error("No API key provided. Use --api-key or set POLYGON_API_KEY environment variable.")
        return 1
    
    # Calculate date range - use historical data (free tier typically allows older data)
    if args.historical:
        end_date = '2022-12-31'  # End of 2022
        start_date = '2022-12-01'  # Start of December 2022
        logger.info("Using historical date range (Dec 2022)")
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    # Initialize PolygonDataSource
    data_source = PolygonDataSource(api_key)
    
    # Fetch data
    logger.info(f"Fetching data for {args.ticker} from {start_date} to {end_date}")
    df = data_source.fetch_historical_data(
        args.ticker, 
        start_date=start_date,
        end_date=end_date,
        timespan=args.timespan
    )
    
    if df is None or df.empty:
        logger.error("Failed to fetch data.")
        return 1
    
    # Display data info
    logger.info(f"Successfully fetched {len(df)} {args.timespan} bars.")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Display some statistics
    logger.info("\nData Summary:")
    logger.info(df.describe())
    
    # Preview the data
    logger.info("\nFirst 5 rows:")
    logger.info(df.head())
    
    # Save to CSV
    output_file = f"{args.ticker}_{args.timespan}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file)
    logger.info(f"Data saved to {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 