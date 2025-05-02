"""
Test script for the dataset management functionality.

This script demonstrates how to use the DataManager's dataset versioning and
management features to check for and download up-to-date datasets.
"""

import sys
import os
import argparse
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_manager import DataManager

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Dataset Management Test')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--tickers', type=str, nargs='+', help='List of tickers to check/update datasets for')
    parser.add_argument('--force-download', action='store_true', help='Force download even if dataset is up-to-date')
    return parser.parse_args()

def main():
    """Main function for dataset management test"""
    args = parse_args()
    
    # Initialize DataManager
    data_manager = DataManager()
    
    # Get tickers from command line
    tickers = args.tickers if args.tickers else [args.ticker]
    
    print(f"Testing dataset management for {len(tickers)} tickers: {', '.join(tickers)}")
    
    for ticker in tickers:
        print(f"\n{'='*40}")
        print(f"Checking dataset for {ticker}...")
        
        # Check if dataset is up-to-date
        is_up_to_date = data_manager.is_dataset_up_to_date(ticker)
        
        if is_up_to_date and not args.force_download:
            print(f"Dataset for {ticker} is up-to-date")
            # Get the dataset
            df = data_manager.get_versioned_dataset(ticker)
            print(f"Dataset covers {df.index[0].date()} to {df.index[-1].date()}")
            print(f"Contains {len(df)} data points")
            
            # Show the latest file path
            latest_file = data_manager.get_latest_dataset_file(ticker)
            print(f"Dataset file: {latest_file}")
        else:
            if args.force_download:
                print(f"Forcing download of new dataset for {ticker}")
            else:
                print(f"Dataset for {ticker} is not up-to-date. Downloading latest data...")
            
            # Download the latest dataset
            start_time = datetime.now()
            df = data_manager.download_latest_dataset(ticker)
            end_time = datetime.now()
            
            if df is not None:
                print(f"Successfully downloaded data for {ticker} in {(end_time - start_time).total_seconds():.2f} seconds")
                print(f"Dataset covers {df.index[0].date()} to {df.index[-1].date()}")
                print(f"Contains {len(df)} data points")
                
                # Show the latest file path
                latest_file = data_manager.get_latest_dataset_file(ticker)
                print(f"Dataset saved to: {latest_file}")
            else:
                print(f"Failed to download data for {ticker}")

if __name__ == '__main__':
    main() 