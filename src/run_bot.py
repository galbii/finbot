#!/usr/bin/env python3

import logging
import os
import pandas as pd
from src.core.bot import EvolvedAppleBot
from src.core.data_fetcher import DataFetcher
from datetime import datetime, timedelta

# Ensure required directories exist before setting up logging
def ensure_data_directory():
    """Ensure required directories exist"""
    for directory in ['logs', 'data', 'plots']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# Create directories first
ensure_data_directory()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'trading_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingBot')

def run_trading_bot(start_date=None, end_date=None, population_size=50, generations=20):
    """
    Run the trading bot with specified parameters.
    
    Args:
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        population_size (int): Size of the population for evolution
        generations (int): Number of generations to evolve
        
    Returns:
        tuple: (best_strategy, performance_metrics, pattern_analysis)
    """
    try:
        # Set default dates if not provided
        if not start_date:
            # Use a date range that matches the available data
            start_date = '2020-01-01'
        if not end_date:
            # Use yesterday as the end date
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"Initializing trading bot for period {start_date} to {end_date}")
        
        # Initialize data fetcher
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            logger.error("ALPHA_VANTAGE_API_KEY environment variable not set")
            return None, None, None
            
        data_fetcher = DataFetcher(api_key=api_key)
        
        # Initialize the bot with data fetcher
        bot = EvolvedAppleBot(start_date=start_date, end_date=end_date, data_fetcher=data_fetcher)
        
        # Fetch and prepare data
        if bot.fetch_data() is None:
            logger.error("Failed to load data. Exiting...")
            return
        
        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        bot.calculate_indicators()
        
        # Evolve trading strategies
        logger.info(f"Starting strategy evolution with population={population_size}, generations={generations}")
        best_strategy = bot.evolve_strategies(population_size=population_size, generations=generations)
        
        if best_strategy is None:
            logger.error("Strategy evolution failed. Exiting...")
            return
        
        # Backtest the best strategy
        logger.info("Backtesting best strategy...")
        backtest_results = bot.backtest_strategy()
        
        if backtest_results is None:
            logger.error("Backtesting failed. Exiting...")
            return
        
        # Analyze and display results
        performance_metrics = bot.analyze_performance()
        
        if performance_metrics:
            logger.info("\nFinal Performance Metrics:")
            for metric, value in performance_metrics.items():
                logger.info(f"{metric}: {value}")
        
        # Analyze strategy patterns
        logger.info("\nAnalyzing strategy patterns...")
        pattern_analysis = bot.analyze_strategy_patterns()
        
        # Generate and save plots
        logger.info("Generating performance plots...")
        bot.plot_results()
        
        logger.info("Trading bot execution completed successfully")
        return best_strategy, performance_metrics, pattern_analysis
        
    except Exception as e:
        logger.error(f"Trading bot execution failed: {str(e)}")
        return None, None, None

def main():
    """Main entry point"""
    try:
        # Run the trading bot with default parameters
        best_strategy, metrics, patterns = run_trading_bot()
        
        if best_strategy:
            logger.info("Trading bot completed successfully")
            logger.info(f"Best strategy: {best_strategy.grammar_string}")
            logger.info(f"Performance metrics: {metrics}")
            logger.info(f"Pattern analysis: {patterns}")
        else:
            logger.error("Trading bot failed to find a valid strategy")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 