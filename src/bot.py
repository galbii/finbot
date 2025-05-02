import logging
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
from typing import Dict, Any, Optional, List, Union, Tuple
import time

from data.data_manager import DataManager
from indicators.indicator_calculator import IndicatorCalculator
from strategy import Strategy
from grammar_parser import GrammarParser
from visualization import PerformanceVisualizer
from logging_system import LoggingSystem
from evolution_manager import EvolutionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Bot')

class Bot:
    """
    Trading strategy optimization bot.
    
    This class is responsible for:
    - Managing historical data for a ticker
    - Creating and evaluating trading strategies
    - Running the grammatical evolution process
    - Tracking strategy history and performance
    """
    
    def __init__(self, ticker, start_date=None, end_date=None, initial_capital=10000):
        """
        Initialize a trading bot for a specific ticker.
        
        Args:
            ticker (str): The stock symbol
            start_date (str, optional): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            initial_capital (float): Initial capital for backtesting
        """
        self.ticker = ticker
        self.historical_data = None
        self.strategies_history = []
        self.best_strategy = None
        self.parameters = {
            'initial_capital': initial_capital,
            'start_date': start_date,
            'end_date': end_date,
            'transaction_cost': 0.001,  # 0.1% per trade
            'slippage': 0.001,  # 0.1% slippage assumption,
            'num_workers': max(1, multiprocessing.cpu_count() - 1),  # Default to CPU count - 1
            'batch_size': 10,  # Strategies per batch for memory efficiency
            'early_termination': True,  # Enable early termination for poor strategies
        }
        
        # Initialize components
        self.data_manager = DataManager()
        self.indicator_calculator = IndicatorCalculator()
        self.grammar_parser = GrammarParser()
        
        # For multiprocessing
        self._data_processed_for_multiprocessing = False
        self._shared_data = None
        
        # Setup
        self._fetch_historical_data()
        self._create_base_strategy()
        
        logger.info(f"Bot initialized for {ticker} with initial capital ${initial_capital}, "
                   f"using {self.parameters['num_workers']} workers for parallel processing")
    
    def _fetch_historical_data(self):
        """Fetch historical OHLCV data for the ticker"""
        logger.info(f"Fetching historical data for {self.ticker}")
        
        try:
            # Check if we have parameters for specific date range
            if self.parameters['start_date'] is not None and self.parameters['end_date'] is not None:
                # If specific dates requested, use direct fetch
                self.historical_data = self.data_manager.fetch_data(
                    self.ticker,
                    start_date=self.parameters['start_date'],
                    end_date=self.parameters['end_date']
                )
            else:
                # Otherwise use the versioned dataset system
                # First check if we have an up-to-date dataset
                if not self.data_manager.is_dataset_up_to_date(self.ticker):
                    logger.info(f"Dataset for {self.ticker} is not up-to-date, downloading latest data")
                    self.historical_data = self.data_manager.download_latest_dataset(self.ticker)
                else:
                    logger.info(f"Using existing up-to-date dataset for {self.ticker}")
                    self.historical_data = self.data_manager.get_versioned_dataset(self.ticker)
            
            if self.historical_data is None or self.historical_data.empty:
                raise ValueError(f"Failed to fetch data for {self.ticker}")
            
            # Optimize memory usage by converting to smaller dtypes where possible
            self._optimize_dataframe_memory()
            
            logger.info(f"Successfully fetched {len(self.historical_data)} data points for {self.ticker}")
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def _optimize_dataframe_memory(self):
        """Optimize memory usage of the historical data DataFrame"""
        if self.historical_data is None:
            return
            
        # Convert float64 columns to float32 to save memory
        for col in self.historical_data.select_dtypes(include=['float64']).columns:
            self.historical_data[col] = self.historical_data[col].astype('float32')
            
        # Convert index to datetime if not already
        if not isinstance(self.historical_data.index, pd.DatetimeIndex):
            self.historical_data.index = pd.to_datetime(self.historical_data.index)
            
        # Ensure column names are standardized
        rename_map = {
            'open': 'Open', 'Open': 'Open',
            'high': 'High', 'High': 'High',
            'low': 'Low', 'Low': 'Low',
            'close': 'Close', 'Close': 'Close',
            'volume': 'Volume', 'Volume': 'Volume',
            'adj close': 'Adj Close', 'Adj Close': 'Adj Close'
        }
        
        self.historical_data = self.historical_data.rename(columns={
            col: rename_map.get(col.lower(), col) 
            for col in self.historical_data.columns
        })
        
        # Log memory usage
        memory_usage = self.historical_data.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"Optimized DataFrame memory usage: {memory_usage:.2f} MB")
    
    def prepare_for_multiprocessing(self):
        """Prepare data for sharing across multiple processes"""
        logger.info("Preparing data for multiprocessing")
        
        # Create shared memory version of the data
        # For now, we'll just make a copy that each process will use
        # In a more advanced implementation, you could use shared memory
        # objects like multiprocessing.Array or Manager dictionaries
        
        # Pre-calculate common indicators to avoid repetitive calculations
        self._precalculate_common_indicators()
        
        # Mark as processed
        self._data_processed_for_multiprocessing = True
        logger.info("Data prepared for multiprocessing")
    
    def _precalculate_common_indicators(self):
        """Pre-calculate commonly used indicators to avoid redundant calculations"""
        logger.info("Pre-calculating common indicators")
        
        # List of common indicators with parameters
        common_indicators = [
            {'name': 'RSI', 'params': {'period': 14}},
            {'name': 'RSI', 'params': {'period': 7}},
            {'name': 'RSI', 'params': {'period': 21}},
            {'name': 'MACD', 'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}},
            {'name': 'BOLLINGER_BANDS', 'params': {'window': 20, 'std': 2}},
            {'name': 'SMA', 'params': {'period': 20}},
            {'name': 'SMA', 'params': {'period': 50}},
            {'name': 'SMA', 'params': {'period': 200}},
            {'name': 'EMA', 'params': {'period': 20}}
        ]
        
        # Batch calculate all indicators
        results = self.indicator_calculator.batch_calculate(
            self.historical_data, 
            common_indicators
        )
        
        logger.info(f"Pre-calculated {len(results)} indicators")
    
    def _create_base_strategy(self):
        """Create initial RSI strategy as baseline"""
        logger.info("Creating base RSI strategy")
        
        # Simple RSI strategy as baseline
        grammar_string = "BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70"
        
        # Create and evaluate base strategy
        base_strategy = Strategy(grammar_string, generation=0)
        
        # Parse the strategy
        if not base_strategy.parse(self.grammar_parser):
            logger.error("Failed to parse base strategy")
            return
        
        # Set as best strategy
        self.best_strategy = base_strategy
        self.strategies_history.append(base_strategy)
        
        logger.info(f"Base strategy created: {grammar_string}")
    
    def run(self, generations=10, population_size=50, mutation_rate=0.2, crossover_rate=0.7, num_workers=None, progress_callback=None):
        """
        Run the grammatical evolution process.
        
        Args:
            generations (int): Number of generations to run
            population_size (int): Size of the population
            mutation_rate (float): Probability of mutation (0-1)
            crossover_rate (float): Probability of crossover (0-1)
            num_workers (int, optional): Number of worker processes to use
            progress_callback (callable, optional): Callback function for progress reporting
                Should accept (generation, best_strategy, avg_fitness, diversity) parameters
        
        Returns:
            Strategy: The best strategy found
        """
        logger.info(f"Starting evolution process with {generations} generations, "
                   f"population size {population_size}")
        
        # Initialize logging system
        logging_system = LoggingSystem()
        logging_system.log_info(f"Starting evolution for {self.ticker} with {generations} generations")
        
        # Use parameters from constructor if not specified
        if num_workers is not None:
            self.parameters['num_workers'] = num_workers
            
        # Create evolution manager
        evolution_manager = EvolutionManager(
            self,
            self.grammar_parser,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            num_workers=self.parameters['num_workers'],
            batch_size=self.parameters['batch_size'],
            early_termination=self.parameters['early_termination']
        )
        
        # Check if we're using adaptive mutation
        if self.parameters.get('adaptive_mutation', False):
            evolution_manager.adaptive_mutation = True
            evolution_manager.min_mutation_rate = self.parameters.get('min_mutation_rate', 0.1)
            evolution_manager.max_mutation_rate = self.parameters.get('max_mutation_rate', 0.5)
            logger.info(f"Using adaptive mutation rates: {evolution_manager.min_mutation_rate} to {evolution_manager.max_mutation_rate}")
            logging_system.log_info(f"Using adaptive mutation rates: {evolution_manager.min_mutation_rate} to {evolution_manager.max_mutation_rate}")
        
        # Check if we're using complexity control
        if self.parameters.get('complexity_control', False):
            evolution_manager.complexity_control = True
            evolution_manager.max_conditions = self.parameters.get('max_conditions', 3)
            logger.info(f"Using complexity control with max {evolution_manager.max_conditions} conditions")
            logging_system.log_info(f"Using complexity control with max {evolution_manager.max_conditions} conditions")
        
        # Prepare for multiprocessing
        if self.parameters['num_workers'] > 1 and not self._data_processed_for_multiprocessing:
            self.prepare_for_multiprocessing()
        
        # Initialize population with best strategy as seed
        evolution_manager.initialize_population(self.best_strategy)
        
        # Evaluate initial population
        evolution_manager.evaluate_population()
        
        # Track metrics across generations for plotting
        generation_metrics = []
        
        # Run evolution for specified number of generations
        for generation in range(1, generations + 1):
            start_time = time.time()
            
            # Evolve population
            evolution_manager.evolve_population()
            
            # Evaluate new population
            evolution_manager.evaluate_population()
            
            # Get best strategy from this generation
            generation_best = evolution_manager.best_strategy
            
            # Update best strategy if this generation's best is better
            if generation_best.profit_percentage > self.best_strategy.profit_percentage:
                generation_best.improvement = (generation_best.profit_percentage - 
                                            self.best_strategy.profit_percentage)
                self.best_strategy = generation_best
                self.strategies_history.append(generation_best)
                
                logger.info(f"Generation {generation}: New best strategy found with "
                           f"{self.best_strategy.profit_percentage:.2f}% profit")
            else:
                logger.info(f"Generation {generation}: No improvement - best remains at "
                           f"{self.best_strategy.profit_percentage:.2f}% profit")
            
            # Calculate population statistics
            avg_fitness = np.mean([s.profit_percentage for s in evolution_manager.population])
            diversity = evolution_manager.calculate_diversity()
            
            # Track metrics
            duration = time.time() - start_time
            metrics = {
                'generation': generation,
                'best_profit': generation_best.profit_percentage,
                'avg_profit': avg_fitness,
                'diversity': diversity,
                'mutation_rate': evolution_manager.mutation_rate,
                'duration': duration
            }
            generation_metrics.append(metrics)
            
            # Log this generation with the logging system
            logging_system.log_evolution_step(
                generation=generation,
                best_strategy=generation_best,
                avg_profit=avg_fitness,
                diversity=diversity,
                mutation_rate=evolution_manager.mutation_rate,
                duration=duration
            )
            
            logger.info(f"Generation {generation} complete in {metrics['duration']:.2f}s - "
                       f"Best: {metrics['best_profit']:.2f}%, "
                       f"Avg: {metrics['avg_profit']:.2f}%, "
                       f"Diversity: {metrics['diversity']:.2f}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(generation, generation_best, avg_fitness, diversity)
        
        # Store generation metrics for visualization
        self.generation_metrics = pd.DataFrame(generation_metrics)
        
        # Log final best strategy performance
        logging_system.log_strategy_performance(self.best_strategy, detailed=True)
        
        # Create evolution report
        report_path = logging_system.create_evolution_report()
        if report_path:
            logger.info(f"Evolution report created: {report_path}")
        
        # Export logs to HTML for easy viewing
        html_report = logging_system.export_logs(format_type='html')
        if html_report:
            logger.info(f"HTML report created: {html_report}")
        
        logger.info(f"Evolution complete - Best strategy: {self.best_strategy.grammar_string}")
        logger.info(f"Final profit: {self.best_strategy.profit_percentage:.2f}%")
        
        return self.best_strategy
    
    def backtest_strategy(self, strategy, early_termination=True):
        """
        Perform backtesting of a strategy on historical data.
        
        Args:
            strategy (Strategy): Strategy to backtest
            early_termination (bool): Whether to enable early termination
            
        Returns:
            dict: Backtesting results
        """
        logger.info(f"Backtesting strategy: {strategy.grammar_string[:50]}...")
        
        # Ensure strategy is parsed
        if strategy.parsed_strategy is None:
            if not strategy.parse(self.grammar_parser):
                logger.error("Failed to parse strategy for backtesting")
                return None
        
        # Import BacktestEngine
        from backtesting import BacktestEngine
        
        # Setup backtest parameters from bot parameters
        backtest_params = {
            'initial_capital': self.parameters['initial_capital'],
            'transaction_cost': self.parameters['transaction_cost'],
            'slippage': self.parameters['slippage']
        }
        
        # Create and run backtesting engine
        engine = BacktestEngine(self.historical_data, self.indicator_calculator, backtest_params)
        results = engine.run_backtest(
            strategy, 
            condition_evaluator=self._evaluate_condition,
            early_termination=early_termination
        )
        
        if results:
            logger.info(f"Backtesting completed: Profit {strategy.profit_percentage:.2f}%, "
                       f"Trades: {len(strategy.trades)}")
            
            # Check if early termination was triggered
            if results.get('early_terminated', False):
                logger.info("Strategy evaluation was terminated early due to poor performance")
            
            # Save equity curve plot if plots directory exists
            if os.path.exists("plots"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = f"plots/equity_curve_{timestamp}.png"
                engine.plot_equity_curve(results['portfolio_history'], save_path=plot_path)
                
                trade_analysis_path = f"plots/trade_analysis_{timestamp}.png"
                engine.plot_trade_analysis(results['trades'], results['portfolio_history'], save_path=trade_analysis_path)
        
        return results
    
    def _evaluate_condition(self, condition, data, idx):
        """
        Evaluate a condition for a specific data point.
        
        Args:
            condition (dict): Condition from parsed strategy
            data (pd.DataFrame): Historical data up to the current point
            idx (int): Index of the current data point
            
        Returns:
            bool: Whether the condition is met
        """
        # For compound conditions with AND/OR
        if isinstance(condition, dict) and 'operator' in condition and 'conditions' in condition:
            if condition['operator'] == 'AND':
                return all(self._evaluate_condition(subcond, data, idx) for subcond in condition['conditions'])
            elif condition['operator'] == 'OR':
                return any(self._evaluate_condition(subcond, data, idx) for subcond in condition['conditions'])
        
        # Single condition
        indicator = condition.get('indicator')
        
        if indicator == 'RSI':
            period = condition['parameters']['period']
            value = condition['value']
            operator = condition['operator']
            
            # Make sure we have enough data
            if idx < period:
                return False
            
            # Calculate RSI
            current_data = data.iloc[:idx+1]
            rsi_values = self.indicator_calculator.calculate_rsi(current_data, period)
            
            if len(rsi_values) < 1:
                return False
                
            current_rsi = rsi_values.iloc[-1]
            
            # Evaluate based on operator
            if operator == '<':
                return current_rsi < value
            elif operator == '>':
                return current_rsi > value
            elif operator == 'CROSSES ABOVE':
                if len(rsi_values) < 2:
                    return False
                return rsi_values.iloc[-2] < value and current_rsi > value
            elif operator == 'CROSSES BELOW':
                if len(rsi_values) < 2:
                    return False
                return rsi_values.iloc[-2] > value and current_rsi < value
        
        elif indicator == 'MACD':
            fast_period = condition['parameters']['fast_period']
            slow_period = condition['parameters']['slow_period']
            signal_period = condition['parameters']['signal_period']
            operator = condition['operator']
            reference = condition['reference']
            
            # Make sure we have enough data
            if idx < max(fast_period, slow_period) + signal_period:
                return False
            
            # Calculate MACD
            current_data = data.iloc[:idx+1]
            macd_values = self.indicator_calculator.calculate_macd(
                current_data, 
                fast_period=fast_period, 
                slow_period=slow_period, 
                signal_period=signal_period
            )
            
            if len(macd_values) < 2:
                return False
                
            # Extract MACD line and signal line
            macd_line = macd_values['macd']
            signal_line = macd_values['signal']
            
            # Evaluate based on operator and reference
            if reference == 'SIGNAL':
                if operator == 'CROSSES ABOVE':
                    return macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
                elif operator == 'CROSSES BELOW':
                    return macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]
            elif reference == '0':
                if operator == 'CROSSES ABOVE':
                    return macd_line.iloc[-2] < 0 and macd_line.iloc[-1] > 0
                elif operator == 'CROSSES BELOW':
                    return macd_line.iloc[-2] > 0 and macd_line.iloc[-1] < 0
        
        elif indicator == 'BOLLINGER_BANDS':
            window = condition['parameters']['window']
            std = condition['parameters']['std']
            operator = condition['operator']
            band = condition['band']
            
            # Make sure we have enough data
            if idx < window:
                return False
            
            # Calculate Bollinger Bands
            current_data = data.iloc[:idx+1]
            bb_values = self.indicator_calculator.calculate_bollinger_bands(
                current_data, window=window, std_dev=std
            )
            
            if len(bb_values) < 2:
                return False
                
            current_price = data.iloc[idx]['Close']
            prev_price = data.iloc[idx-1]['Close']
            
            if band == 'UPPER':
                upper_band = bb_values['upper']
                if operator == '>':
                    return current_price > upper_band.iloc[-1]
                elif operator == '<':
                    return current_price < upper_band.iloc[-1]
                elif operator == 'CROSSES ABOVE':
                    return prev_price < upper_band.iloc[-2] and current_price > upper_band.iloc[-1]
                elif operator == 'CROSSES BELOW':
                    return prev_price > upper_band.iloc[-2] and current_price < upper_band.iloc[-1]
            elif band == 'LOWER':
                lower_band = bb_values['lower']
                if operator == '>':
                    return current_price > lower_band.iloc[-1]
                elif operator == '<':
                    return current_price < lower_band.iloc[-1]
                elif operator == 'CROSSES ABOVE':
                    return prev_price < lower_band.iloc[-2] and current_price > lower_band.iloc[-1]
                elif operator == 'CROSSES BELOW':
                    return prev_price > lower_band.iloc[-2] and current_price < lower_band.iloc[-1]
        
        elif indicator == 'MA_CROSS':
            params = condition['parameters']
            ma1_type = params['ma1_type']
            ma1_period = params['ma1_period']
            ma2_type = params['ma2_type']
            ma2_period = params['ma2_period']
            operator = condition['operator']
            
            # Make sure we have enough data
            if idx < max(ma1_period, ma2_period):
                return False
            
            # Calculate moving averages
            current_data = data.iloc[:idx+1]
            
            # Calculate MA1
            if ma1_type == 'SMA':
                ma1_values = self.indicator_calculator.calculate_sma(current_data, ma1_period)
            else:  # EMA
                ma1_values = self.indicator_calculator.calculate_ema(current_data, ma1_period)
            
            # Calculate MA2
            if ma2_type == 'SMA':
                ma2_values = self.indicator_calculator.calculate_sma(current_data, ma2_period)
            else:  # EMA
                ma2_values = self.indicator_calculator.calculate_ema(current_data, ma2_period)
            
            if len(ma1_values) < 2 or len(ma2_values) < 2:
                return False
            
            # Evaluate based on operator
            if operator == 'CROSSES ABOVE':
                return (ma1_values.iloc[-2] < ma2_values.iloc[-2] and 
                        ma1_values.iloc[-1] > ma2_values.iloc[-1])
            elif operator == 'CROSSES BELOW':
                return (ma1_values.iloc[-2] > ma2_values.iloc[-2] and 
                        ma1_values.iloc[-1] < ma2_values.iloc[-1])
        
        # Default fallback
        logger.warning(f"Unsupported condition type: {indicator}")
        return False
    
    def _calculate_position_size(self, position_sizing, available_cash, price):
        """
        Calculate position size based on position sizing rule.
        
        Args:
            position_sizing (dict): Position sizing rule
            available_cash (float): Available cash in portfolio
            price (float): Current price
            
        Returns:
            float: Number of shares to buy
        """
        if position_sizing['type'] == 'FIXED':
            # Fixed dollar amount
            return min(position_sizing['amount'] / price, available_cash / price)
        
        elif position_sizing['type'] == 'PERCENTAGE':
            # Percentage of available cash
            percentage = position_sizing['percentage'] / 100
            return (available_cash * percentage) / price
        
        elif position_sizing['type'] == 'RISK_BASED':
            # Risk-based position sizing (simplified)
            risk_percentage = position_sizing['risk_percentage'] / 100
            return (available_cash * risk_percentage) / price
        
        # Default to using all available cash
        return available_cash / price
    
    def view(self, detailed=False):
        """
        Display results of the optimization process.
        
        Args:
            detailed (bool): Whether to show detailed results
            
        Returns:
            dict: Results summary
        """
        if not self.best_strategy:
            logger.warning("No strategies have been evaluated yet")
            return None
        
        logger.info(f"Best strategy found (Generation {self.best_strategy.generation}):")
        logger.info(f"Strategy: {self.best_strategy.grammar_string}")
        logger.info(f"Profit: {self.best_strategy.profit_percentage:.2f}%")
        logger.info(f"Sharpe Ratio: {self.best_strategy.metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {self.best_strategy.metrics['max_drawdown']:.2f}%")
        logger.info(f"Win Rate: {self.best_strategy.metrics['win_rate']:.2f}%")
        
        if detailed and self.best_strategy.trades:
            logger.info(f"Number of trades: {len(self.best_strategy.trades)}")
            for trade in self.best_strategy.trades[:5]:  # Show just first 5 trades as example
                logger.info(f"Trade: {trade}")
            
            if len(self.best_strategy.trades) > 5:
                logger.info("... (more trades) ...")
        
        return {
            'best_strategy': self.best_strategy.to_dict(),
            'generations': len(set(s.generation for s in self.strategies_history)),
            'strategies_evaluated': len(self.strategies_history)
        }
    
    def save(self, filename=None):
        """
        Save the bot state to a file.
        
        Args:
            filename (str, optional): Filename to save to
            
        Returns:
            str: Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.ticker}_bot_{timestamp}.json"
        
        data = {
            'ticker': self.ticker,
            'parameters': self.parameters,
            'best_strategy': self.best_strategy.to_dict() if self.best_strategy else None,
            'strategies_history': [s.to_dict() for s in self.strategies_history]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Bot state saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save bot state: {e}")
            return None
    
    @classmethod
    def load(cls, filename):
        """
        Load a bot from a saved file.
        
        Args:
            filename (str): Path to saved bot file
            
        Returns:
            Bot: Loaded bot instance
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Create bot instance
            bot = cls(
                data['ticker'],
                start_date=data['parameters']['start_date'],
                end_date=data['parameters']['end_date'],
                initial_capital=data['parameters']['initial_capital']
            )
            
            # Override parameters
            bot.parameters = data['parameters']
            
            # Load strategies history
            bot.strategies_history = [
                Strategy.from_dict(s_data) for s_data in data['strategies_history']
            ]
            
            # Load best strategy
            if data['best_strategy']:
                bot.best_strategy = Strategy.from_dict(data['best_strategy'])
            
            logger.info(f"Bot loaded from {filename}")
            return bot
        except Exception as e:
            logger.error(f"Failed to load bot from {filename}: {e}")
            return None
    
    def delete(self):
        """Clean up resources"""
        logger.info("Cleaning up bot resources")
        
        # Clear caches to free memory
        if hasattr(self, 'data_manager'):
            self.data_manager.clear_cache()
        
        if hasattr(self, 'indicator_calculator'):
            self.indicator_calculator.clear_cache()
        
        # Clear references to large objects
        self.historical_data = None
        self.strategies_history = []
        self.best_strategy = None
        
        logger.info("Bot resources cleaned up")
    
    def visualize_performance(self):
        """
        Create performance visualization charts for the best strategy.
        
        Creates:
        - Equity curve
        - Drawdown chart
        - Monthly returns heatmap
        - Performance metrics summary
        
        Returns:
            str: Directory path where images are saved
        """
        logger.info("Generating performance visualizations")
        
        # Create visualizer
        visualizer = PerformanceVisualizer()
        
        # Ensure best strategy has portfolio history
        if not hasattr(self.best_strategy, 'portfolio_history') or self.best_strategy.portfolio_history is None:
            logger.info("Running backtest to generate portfolio history")
            self.backtest_strategy(self.best_strategy)
        
        # Define output paths
        plots_dir = visualizer.output_dir
        equity_path = os.path.join(plots_dir, f"{self.ticker}_equity_curve.png")
        drawdown_path = os.path.join(plots_dir, f"{self.ticker}_drawdown.png")
        monthly_path = os.path.join(plots_dir, f"{self.ticker}_monthly_returns.png")
        report_path = os.path.join(plots_dir, f"{self.ticker}_performance_report.html")
        
        # Generate visualizations
        visualizer.plot_equity_curve(
            self.best_strategy.portfolio_history,
            title=f"{self.ticker} Equity Curve - {self.best_strategy.profit_percentage:.2f}% Profit",
            save_path=equity_path
        )
        
        visualizer.plot_drawdown_chart(
            self.best_strategy.portfolio_history,
            title=f"{self.ticker} Drawdown Chart",
            save_path=drawdown_path
        )
        
        visualizer.plot_monthly_returns_heatmap(
            self.best_strategy.portfolio_history,
            title=f"{self.ticker} Monthly Returns",
            save_path=monthly_path
        )
        
        visualizer.generate_performance_report(
            self.best_strategy,
            strategy_name=f"{self.ticker} - Best Strategy",
            save_path=report_path
        )
        
        logger.info(f"Performance visualizations saved to {plots_dir}")
        return plots_dir
    
    def visualize_trades(self):
        """
        Create trade visualization charts.
        
        Creates:
        - Trade entry/exit points on price chart
        - Trade profit/loss distribution
        - Win/loss streaks
        
        Returns:
            str: Directory path where images are saved
        """
        logger.info("Generating trade visualizations")
        
        # Create visualizer
        visualizer = PerformanceVisualizer()
        
        # Ensure best strategy has trades
        if not hasattr(self.best_strategy, 'trades') or not self.best_strategy.trades:
            logger.info("Running backtest to generate trade data")
            self.backtest_strategy(self.best_strategy)
        
        # Define output path
        plots_dir = visualizer.output_dir
        trades_path = os.path.join(plots_dir, f"{self.ticker}_trades.png")
        
        # Generate visualization
        visualizer.plot_trade_analysis(
            self.best_strategy.trades,
            title=f"{self.ticker} Trade Analysis",
            save_path=trades_path
        )
        
        logger.info(f"Trade visualizations saved to {plots_dir}")
        return plots_dir
    
    def visualize_evolution_progress(self):
        """
        Create evolution progress charts.
        
        Creates:
        - Strategy improvement over generations
        - Population diversity over generations
        - Average fitness over generations
        
        Returns:
            str: Directory path where images are saved
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        logger.info("Generating evolution progress visualizations")
        
        # Check if we have generation metrics
        if not hasattr(self, 'generation_metrics') or self.generation_metrics is None or len(self.generation_metrics) == 0:
            logger.warning("No generation metrics available. Run evolution first.")
            return None
        
        # Create output directory if it doesn't exist
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Set style
        sns.set_style('whitegrid')
        
        # Create figure with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot 1: Best and average profit percentage
        axs[0].plot(self.generation_metrics['generation'], self.generation_metrics['best_profit'], 
                   'b-', linewidth=2, label='Best Strategy')
        axs[0].plot(self.generation_metrics['generation'], self.generation_metrics['avg_profit'], 
                   'r--', linewidth=1.5, label='Population Average')
        axs[0].set_ylabel('Profit %')
        axs[0].set_title('Strategy Performance Evolution')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot 2: Population diversity
        axs[1].plot(self.generation_metrics['generation'], self.generation_metrics['diversity'], 
                   'g-', linewidth=2)
        axs[1].set_ylabel('Diversity')
        axs[1].set_title('Population Diversity')
        axs[1].grid(True)
        
        # Plot 3: Generation duration
        axs[2].bar(self.generation_metrics['generation'], self.generation_metrics['duration'], 
                  color='purple', alpha=0.7)
        axs[2].set_xlabel('Generation')
        axs[2].set_ylabel('Time (seconds)')
        axs[2].set_title('Generation Duration')
        axs[2].grid(True)
        
        # Set x-axis ticks to integers
        axs[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        # Save figure
        evolution_path = os.path.join(plots_dir, f"{self.ticker}_evolution_progress.png")
        plt.savefig(evolution_path)
        plt.close()
        
        logger.info(f"Evolution progress visualization saved to {evolution_path}")
        return plots_dir
    
    def visualize_strategy_comparison(self):
        """
        Create strategy comparison charts.
        
        Creates:
        - Equity curves of top strategies
        - Strategy complexity vs performance
        - Strategy parameters comparison
        
        Returns:
            str: Directory path where images are saved
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import ListedColormap
        
        logger.info("Generating strategy comparison visualizations")
        
        # Check if we have enough strategies to compare
        if len(self.strategies_history) < 2:
            logger.warning("Not enough strategies to compare. Need at least 2.")
            return None
        
        # Create output directory if it doesn't exist
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Set style
        sns.set_style('whitegrid')
        
        # 1. Top Strategies Comparison
        # Select top 5 strategies or all if less than 5
        top_n = min(5, len(self.strategies_history))
        top_strategies = sorted(self.strategies_history, 
                               key=lambda s: s.profit_percentage, 
                               reverse=True)[:top_n]
        
        # Ensure all top strategies have portfolio history
        for strategy in top_strategies:
            if not hasattr(strategy, 'portfolio_history') or strategy.portfolio_history is None:
                self.backtest_strategy(strategy)
        
        # Create figure for equity curves
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve for each strategy
        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        
        for i, strategy in enumerate(top_strategies):
            plt.plot(strategy.portfolio_history.index, 
                    strategy.portfolio_history['portfolio_value'], 
                    label=f"Gen {strategy.generation} - {strategy.profit_percentage:.2f}%", 
                    color=colors[i], linewidth=2)
        
        plt.title(f"{self.ticker} - Top {top_n} Strategies Comparison")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        comparison_path = os.path.join(plots_dir, f"{self.ticker}_strategy_comparison.png")
        plt.savefig(comparison_path)
        plt.close()
        
        # 2. Strategy Complexity vs Performance
        plt.figure(figsize=(10, 6))
        
        # Calculate complexity (number of conditions)
        complexities = []
        profits = []
        generations = []
        
        for strategy in self.strategies_history:
            # Simple complexity measure: count the number of conditions
            complexity = strategy.grammar_string.count('AND') + strategy.grammar_string.count('OR') + 1
            complexities.append(complexity)
            profits.append(strategy.profit_percentage)
            generations.append(strategy.generation)
        
        # Create scatter plot with color based on generation
        plt.scatter(complexities, profits, c=generations, cmap='viridis', 
                   s=100, alpha=0.7, edgecolors='k', linewidths=1)
        
        plt.colorbar(label='Generation')
        plt.title(f"{self.ticker} - Strategy Complexity vs Performance")
        plt.xlabel("Strategy Complexity (Number of Conditions)")
        plt.ylabel("Profit %")
        plt.grid(True)
        
        # Save figure
        complexity_path = os.path.join(plots_dir, f"{self.ticker}_complexity_vs_performance.png")
        plt.savefig(complexity_path)
        plt.close()
        
        logger.info(f"Strategy comparison visualizations saved to {plots_dir}")
        return plots_dir 