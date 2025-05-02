import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

from data.data_manager import DataManager
from indicators.indicator_calculator import IndicatorCalculator
from bot import Bot
from strategy import Strategy
from grammar_parser import GrammarParser
from visualization import PerformanceVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingSystem')

class ProgressReporter:
    """Handles progress reporting for long-running operations"""
    
    def __init__(self, total, description="Processing"):
        self.pbar = tqdm(total=total, desc=description)
        self.last_update = 0
        
    def update(self, current):
        """Update progress bar with current progress"""
        increment = current - self.last_update
        if increment > 0:
            self.pbar.update(increment)
            self.last_update = current
            
    def close(self):
        """Close the progress bar"""
        self.pbar.close()

def validate_ticker(ticker):
    """Validate ticker symbol format"""
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker symbol must be a non-empty string")
    if not ticker.isalnum():
        raise ValueError("Ticker symbol must contain only alphanumeric characters")
    return ticker.upper()

def validate_date(date_str, allow_none=True):
    """Validate date string format YYYY-MM-DD"""
    if date_str is None and allow_none:
        return None
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD")

def validate_positive_float(value, param_name):
    """Validate positive float value"""
    try:
        float_val = float(value)
        if float_val <= 0:
            raise ValueError
        return float_val
    except (ValueError, TypeError):
        raise ValueError(f"{param_name} must be a positive number")

def validate_positive_int(value, param_name):
    """Validate positive integer value"""
    try:
        int_val = int(value)
        if int_val <= 0:
            raise ValueError
        return int_val
    except (ValueError, TypeError):
        raise ValueError(f"{param_name} must be a positive integer")

def validate_percentage(value, param_name):
    """Validate percentage value between 0 and 1"""
    try:
        float_val = float(value)
        if float_val < 0 or float_val > 1:
            raise ValueError
        return float_val
    except (ValueError, TypeError):
        raise ValueError(f"{param_name} must be a value between 0 and 1")

def create_bot(args):
    """Create a new trading bot with validation"""
    try:
        # Validate parameters
        ticker = validate_ticker(args.ticker)
        start_date = validate_date(args.start_date)
        end_date = validate_date(args.end_date)
        initial_capital = validate_positive_float(args.initial_capital, "Initial capital")
        
        # If dates are not specified, calculate based on days
        if start_date is None and args.days:
            days = validate_positive_int(args.days, "Days")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date_obj = datetime.now() - timedelta(days=days)
            start_date = start_date_obj.strftime('%Y-%m-%d')
        
        logger.info(f"Creating bot for {ticker} from {start_date} to {end_date}")
        
        # Create progress reporter
        progress = ProgressReporter(4, "Creating bot")
        
        # Step 1: Initialize bot
        progress.update(1)
        bot = Bot(ticker, start_date=start_date, end_date=end_date, initial_capital=initial_capital)
        
        # Step 2: Save bot state if specified
        progress.update(2)
        if args.save_bot:
            bot.save(args.save_bot)
            logger.info(f"Bot saved to {args.save_bot}")
        
        # Step 3: Bot created successfully
        progress.update(4)
        progress.close()
        
        logger.info(f"Bot created successfully for {ticker}")
        print(f"Successfully created bot for {ticker}")
        print(f"Historical data loaded: {len(bot.historical_data)} data points")
        print(f"Date range: {bot.historical_data.index[0].date()} to {bot.historical_data.index[-1].date()}")
        
        # Clean up if not required anymore
        if not args.run_evolution:
            bot.delete()
        
        return bot
    except Exception as e:
        logger.error(f"Error creating bot: {str(e)}")
        print(f"Error: {str(e)}")
        return None

def run_evolution(args, bot=None):
    """Run the evolution process with validation and progress reporting"""
    try:
        # Load bot if not provided
        if bot is None and args.load_bot:
            logger.info(f"Loading bot from {args.load_bot}")
            bot = Bot.load(args.load_bot)
            if not bot:
                raise ValueError(f"Failed to load bot from {args.load_bot}")
        
        if bot is None:
            raise ValueError("No bot provided. Create a bot first or specify a bot file to load")
        
        # Validate parameters
        generations = validate_positive_int(args.generations, "Generations")
        population_size = validate_positive_int(args.population_size, "Population size")
        mutation_rate = validate_percentage(args.mutation_rate, "Mutation rate")
        crossover_rate = validate_percentage(args.crossover_rate, "Crossover rate")
        
        # Set adaptive mutation parameters if enabled
        if args.adaptive_mutation:
            bot.parameters['adaptive_mutation'] = True
            bot.parameters['min_mutation_rate'] = 0.1
            bot.parameters['max_mutation_rate'] = 0.5
        
        # Set complexity control if enabled
        if args.complexity_control:
            bot.parameters['complexity_control'] = True
            bot.parameters['max_conditions'] = 3
        
        logger.info(f"Running evolution with {generations} generations, population size {population_size}")
        
        # Create a callback for progress reporting
        class EvolutionProgressCallback:
            def __init__(self, generations):
                self.progress = ProgressReporter(generations, "Evolution progress")
                self.last_gen = 0
                self.best_profit = 0
                
            def __call__(self, gen, best_strategy, avg_fitness, diversity):
                self.progress.update(gen)
                self.last_gen = gen
                profit = best_strategy.profit_percentage
                self.best_profit = profit
                
                # Display current generation stats
                print(f"\nGeneration {gen}/{generations}:")
                print(f"Best strategy: {best_strategy.profit_percentage:.2f}% profit")
                print(f"Avg fitness: {avg_fitness:.2f}%, Diversity: {diversity:.2f}")
                
            def close(self):
                self.progress.close()
                
        # Run evolution with progress callback
        progress_callback = EvolutionProgressCallback(generations)
        
        # Run evolution
        bot.run(
            generations=generations, 
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            progress_callback=progress_callback
        )
        
        progress_callback.close()
        
        logger.info("Evolution process completed")
        print("\nEvolution completed successfully!")
        print(f"Best strategy found: {bot.best_strategy.profit_percentage:.2f}% profit")
        
        # Save bot if specified
        if args.save_bot:
            bot.save(args.save_bot)
            logger.info(f"Bot saved to {args.save_bot}")
            print(f"Bot state saved to {args.save_bot}")
        
        return bot
    except Exception as e:
        logger.error(f"Error running evolution: {str(e)}")
        print(f"Error: {str(e)}")
        return None

def view_bot(args):
    """View bot results with validation"""
    try:
        # Load bot
        if not args.load_bot:
            raise ValueError("You must specify a bot file to load using --load-bot")
        
        logger.info(f"Loading bot from {args.load_bot}")
        bot = Bot.load(args.load_bot)
        if not bot:
            raise ValueError(f"Failed to load bot from {args.load_bot}")
        
        # Prepare visualizer
        visualizer = PerformanceVisualizer()
        
        # Display detailed results if requested
        print("\nStrategy Results:")
        bot.view(detailed=args.detailed)
        
        # Generate visualizations based on visualization type
        viz_types = args.visualization_types.split(',') if args.visualization_types else ['basic']
        
        # Progress reporting
        progress = ProgressReporter(len(viz_types), "Generating visualizations")
        
        for i, viz_type in enumerate(viz_types):
            viz_type = viz_type.strip().lower()
            
            if viz_type == 'basic' or viz_type == 'all':
                # Create basic performance charts
                bot.visualize_performance()
            
            if viz_type == 'trades' or viz_type == 'all':
                # Create trade visualization
                bot.visualize_trades()
            
            if viz_type == 'evolution' or viz_type == 'all':
                # Create evolution progress charts
                bot.visualize_evolution_progress()
            
            if viz_type == 'comparison' or viz_type == 'all':
                # Create strategy comparison
                bot.visualize_strategy_comparison()
            
            progress.update(i+1)
        
        progress.close()
        print("\nVisualization complete. Results saved to the 'plots' directory.")
        
        # Clean up
        bot.delete()
        return True
    except Exception as e:
        logger.error(f"Error viewing bot: {str(e)}")
        print(f"Error: {str(e)}")
        return False

def delete_bot(args):
    """Delete bot files with validation"""
    try:
        if not args.delete_bot:
            raise ValueError("You must specify a bot file to delete")
        
        # Validate that file exists
        if not os.path.exists(args.delete_bot):
            raise ValueError(f"Bot file not found: {args.delete_bot}")
        
        # Confirm deletion
        if not args.force and input(f"Are you sure you want to delete {args.delete_bot}? (y/n): ").lower() != 'y':
            print("Deletion cancelled.")
            return False
        
        # Delete file
        os.remove(args.delete_bot)
        logger.info(f"Deleted bot file: {args.delete_bot}")
        print(f"Successfully deleted bot file: {args.delete_bot}")
        return True
    except Exception as e:
        logger.error(f"Error deleting bot: {str(e)}")
        print(f"Error: {str(e)}")
        return False

def setup_parser():
    """Setup command line argument parser with subcommands"""
    parser = argparse.ArgumentParser(description='Trading Strategy System')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create bot command
    create_parser = subparsers.add_parser('createbot', help='Create a new trading bot')
    create_parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    create_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    create_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    create_parser.add_argument('--days', type=int, default=365, help='Number of days to fetch if start date not provided')
    create_parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital for backtesting')
    create_parser.add_argument('--save-bot', type=str, help='Save bot state to the specified file')
    create_parser.add_argument('--run-evolution', action='store_true', help='Run evolution after creating bot')
    
    # Run evolution command
    run_parser = subparsers.add_parser('run', help='Run the evolution process')
    run_parser.add_argument('--load-bot', type=str, help='Load bot state from the specified file')
    run_parser.add_argument('--generations', type=int, default=10, help='Number of generations for evolution')
    run_parser.add_argument('--population-size', type=int, default=50, help='Population size for evolution')
    run_parser.add_argument('--mutation-rate', type=float, default=0.2, help='Mutation rate for evolution')
    run_parser.add_argument('--crossover-rate', type=float, default=0.7, help='Crossover rate for evolution')
    run_parser.add_argument('--save-bot', type=str, help='Save bot state to the specified file')
    run_parser.add_argument('--adaptive-mutation', action='store_true', help='Enable adaptive mutation rates')
    run_parser.add_argument('--complexity-control', action='store_true', help='Enable strategy complexity control')
    
    # View results command
    view_parser = subparsers.add_parser('view', help='View bot results')
    view_parser.add_argument('--load-bot', type=str, required=True, help='Load bot state from the specified file')
    view_parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    view_parser.add_argument('--visualization-types', type=str, 
                          help='Types of visualizations to generate (comma-separated): basic,trades,evolution,comparison,all')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a bot file')
    delete_parser.add_argument('--delete-bot', type=str, required=True, help='Bot file to delete')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    
    return parser

def main():
    """Main function with improved command interface"""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Default to help if no command provided
    if not args.command:
        parser.print_help()
        return
    
    # Execute appropriate command
    if args.command == 'createbot':
        bot = create_bot(args)
        if bot and args.run_evolution:
            run_evolution(args, bot)
    elif args.command == 'run':
        run_evolution(args)
    elif args.command == 'view':
        view_bot(args)
    elif args.command == 'delete':
        delete_bot(args)

if __name__ == '__main__':
    main() 