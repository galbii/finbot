import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from src.evolution.evolution_manager import EvolutionManager
from src.evolution.grammar_parser import GrammarParser
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import logging
from .strategy import Strategy
from .data_fetcher import DataFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EvolvedAppleBot')

class EvolvedAppleBot:
    def __init__(self, start_date='2020-01-01', end_date=None, data_fetcher=None):
        """
        Initialize the trading bot.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to yesterday.
            data_fetcher (DataFetcher, optional): DataFetcher instance for fetching data
        """
        self.ticker = 'AAPL'
        
        # Validate and set dates
        try:
            self.start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            if end_date:
                self.end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            else:
                self.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Validate date range
            if pd.to_datetime(self.start_date) >= pd.to_datetime(self.end_date):
                raise ValueError("Start date must be before end date")
            
            if pd.to_datetime(self.end_date) > datetime.now():
                raise ValueError("End date cannot be in the future")
                
        except Exception as e:
            logger.error(f"Invalid date format: {e}")
            raise
        
        self.data = None
        self.indicators = {}
        self.best_strategy = None
        self.evolution_manager = None
        self.grammar_parser = GrammarParser()
        self.data_fetcher = data_fetcher
        
    def fetch_data(self):
        """Fetch historical data for the specified ticker"""
        try:
            if self.data_fetcher is None:
                logger.error("DataFetcher not initialized. Please provide a DataFetcher instance.")
                return None
                
            # Fetch data using DataFetcher
            logger.info(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}")
            self.data = self.data_fetcher.get_historical_data(
                symbol=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if self.data is None or self.data.empty:
                logger.error("No data fetched from Alpha Vantage")
                return None
                
            # Rename columns to match our expected format
            self.data = self.data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Set date as index
            self.data.set_index('date', inplace=True)
            
            # Remove any rows with NaN values
            self.data.dropna(inplace=True)
            
            # Verify we have enough data
            if len(self.data) < 400:  # Need at least 400 days for all indicators
                logger.error(f"Insufficient data points: {len(self.data)}")
                return None
            
            # Calculate indicators
            self.calculate_indicators()
            
            logger.info(f"Successfully loaded {len(self.data)} data points")
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return None
    
    def calculate_indicators(self):
        """Calculate all possible technical indicators for strategy evolution"""
        if self.data is None or self.data.empty:
            logger.error("No data available. Please fetch data first.")
            return
        
        try:
            # RSI with different periods
            for period in [7, 9, 14, 21, 30]:
                rsi = RSIIndicator(close=self.data['Close'], window=period)
                self.data[f'RSI_{period}'] = rsi.rsi()
                logger.debug(f"Calculated RSI with period {period}")
            
            # Moving Averages
            for period in [10, 20, 50, 100, 200]:
                sma = SMAIndicator(close=self.data['Close'], window=period)
                ema = EMAIndicator(close=self.data['Close'], window=period)
                self.data[f'SMA_{period}'] = sma.sma_indicator()
                self.data[f'EMA_{period}'] = ema.ema_indicator()
                logger.debug(f"Calculated MA with period {period}")
            
            # Bollinger Bands with different parameters
            for window in [10, 15, 20, 30]:  # Added 15 to match the error message
                for std in [1.5, 2.0, 2.5, 3.0]:  # Added 3.0 to match the grammar parser
                    try:
                        bb = BollingerBands(close=self.data['Close'], window=window, window_dev=std)
                        bb_upper = bb.bollinger_hband()
                        bb_lower = bb.bollinger_lband()
                        bb_middle = bb.bollinger_mavg()
                        
                        # Format column names to match what's expected in evaluate_single_condition
                        self.data[f'BB_upper_{window}_{std}'] = bb_upper
                        self.data[f'BB_lower_{window}_{std}'] = bb_lower
                        self.data[f'BB_middle_{window}_{std}'] = bb_middle
                        
                        # Fill NaN values that might occur at the start
                        self.data[f'BB_upper_{window}_{std}'].fillna(method='bfill', inplace=True)
                        self.data[f'BB_lower_{window}_{std}'].fillna(method='bfill', inplace=True)
                        self.data[f'BB_middle_{window}_{std}'].fillna(method='bfill', inplace=True)
                        
                        logger.debug(f"Calculated Bollinger Bands with window {window} and std {std}")
                    except Exception as e:
                        logger.error(f"Error calculating Bollinger Bands for window={window}, std={std}: {str(e)}")
                        raise
            
            # VWAP
            vwap = VolumeWeightedAveragePrice(
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                volume=self.data['Volume']
            )
            self.data['VWAP'] = vwap.volume_weighted_average_price()
            logger.debug("Calculated VWAP")
            
            # Calculate MACD for different parameter combinations
            for fast_period in [8, 12, 16, 20]:
                for slow_period in [21, 26, 30, 35]:
                    for signal_period in [5, 7, 9, 12]:
                        # Calculate MACD components
                        exp1 = self.data['Close'].ewm(span=fast_period, adjust=False).mean()
                        exp2 = self.data['Close'].ewm(span=slow_period, adjust=False).mean()
                        macd = exp1 - exp2
                        signal = macd.ewm(span=signal_period, adjust=False).mean()
                        
                        # Store MACD and signal line with consistent naming
                        macd_key = f'MACD_{fast_period}_{slow_period}_{signal_period}'
                        signal_key = f'MACD_SIGNAL_{fast_period}_{slow_period}_{signal_period}'
                        
                        self.data[macd_key] = macd
                        self.data[signal_key] = signal
                        logger.debug(f"Calculated MACD with parameters {fast_period}, {slow_period}, {signal_period}")
            
            # Fill any NaN values that might have been created
            self.data.fillna(method='bfill', inplace=True)
            self.data.fillna(method='ffill', inplace=True)
            
            # Verify all indicators are present
            self._verify_indicators()
            
            logger.info("All indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
    
    def _verify_indicators(self):
        """Verify that all necessary indicators are present in the data"""
        try:
            # Check RSI columns
            for period in [7, 9, 14, 21, 30]:
                if f'RSI_{period}' not in self.data.columns:
                    raise ValueError(f"Missing RSI indicator with period {period}")
            
            # Check MA columns
            for period in [10, 20, 50, 100, 200]:
                if f'SMA_{period}' not in self.data.columns:
                    raise ValueError(f"Missing SMA indicator with period {period}")
                if f'EMA_{period}' not in self.data.columns:
                    raise ValueError(f"Missing EMA indicator with period {period}")
            
            # Check Bollinger Bands columns
            for window in [10, 15, 20, 30]:
                for std in [1.5, 2.0, 2.5, 3.0]:  # Added 3.0 to match the grammar parser
                    if f'BB_upper_{window}_{std}' not in self.data.columns:
                        raise ValueError(f"Missing upper Bollinger Band with window {window} and std {std}")
                    if f'BB_lower_{window}_{std}' not in self.data.columns:
                        raise ValueError(f"Missing lower Bollinger Band with window {window} and std {std}")
                    if f'BB_middle_{window}_{std}' not in self.data.columns:
                        raise ValueError(f"Missing middle Bollinger Band with window {window} and std {std}")
            
            # Check MACD columns
            for fast_period in [8, 12, 16, 20]:
                for slow_period in [21, 26, 30, 35]:
                    for signal_period in [5, 7, 9, 12]:
                        macd_key = f'MACD_{fast_period}_{slow_period}_{signal_period}'
                        signal_key = f'MACD_SIGNAL_{fast_period}_{slow_period}_{signal_period}'
                        if macd_key not in self.data.columns:
                            raise ValueError(f"Missing MACD line with parameters {fast_period}, {slow_period}, {signal_period}")
                        if signal_key not in self.data.columns:
                            raise ValueError(f"Missing MACD signal with parameters {fast_period}, {slow_period}, {signal_period}")
            
            # Check VWAP
            if 'VWAP' not in self.data.columns:
                raise ValueError("Missing VWAP indicator")
                
            logger.info("All required indicators verified")
            
        except Exception as e:
            logger.error(f"Error verifying indicators: {str(e)}")
            raise
    
    def evolve_strategies(self, population_size=50, generations=20):
        """Evolve trading strategies using grammatical evolution"""
        if self.data is None:
            logger.error("No data available. Please fetch data first.")
            return
        
        try:
            # Validate population size and generations
            if population_size < 10:
                raise ValueError("Population size must be at least 10")
            if generations < 5:
                raise ValueError("Number of generations must be at least 5")
            
            # Initialize evolution manager
            self.evolution_manager = EvolutionManager(
                bot=self,
                grammar_parser=self.grammar_parser,
                population_size=population_size,
                mutation_rate=0.3,
                crossover_rate=0.7,
                tournament_size=3,
                elitism_count=2
            )
            
            # Initialize population with a simple RSI strategy as base
            base_strategy = Strategy("BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70", generation=0)
            self.evolution_manager.initialize_population(base_strategy)
            
            logger.info(f"Starting evolution with population size {population_size}")
            
            # Track best fitness history
            best_fitness_history = []
            
            # Run evolution for specified generations
            for gen in range(generations):
                logger.info(f"Generation {gen + 1}/{generations}")
                
                try:
                    # Evaluate current population
                    self.evolution_manager.evaluate_population()
                    
                    # Get best strategy of this generation
                    gen_best = max(self.evolution_manager.population, key=lambda x: x.fitness)
                    best_fitness_history.append(gen_best.fitness)
                    
                    logger.info(f"Generation {gen + 1} best fitness: {gen_best.fitness:.4f}")
                    logger.info(f"Best strategy: {gen_best.grammar_string}")
                    
                    # Update overall best strategy if needed
                    if self.best_strategy is None or gen_best.fitness > self.best_strategy.fitness:
                        self.best_strategy = gen_best.copy()
                        logger.info(f"New best strategy found! Fitness: {self.best_strategy.fitness:.4f}")
                    
                    # Check for convergence
                    if len(best_fitness_history) >= 5:
                        recent_improvement = best_fitness_history[-1] - best_fitness_history[-5]
                        if recent_improvement < 0.01:  # Less than 1% improvement in last 5 generations
                            logger.info("Convergence detected. Stopping evolution early.")
                            break
                    
                    # Get fitness scores for current population
                    fitness_scores = [s.fitness for s in self.evolution_manager.population]
                    
                    # Evolve population
                    self.evolution_manager.population = self.evolution_manager.evolve_population(
                        self.evolution_manager.population,
                        fitness_scores
                    )
                    
                except Exception as e:
                    logger.error(f"Error in generation {gen + 1}: {str(e)}")
                    continue
            
            if self.best_strategy is None:
                raise ValueError("No valid strategies were generated during evolution")
            
            logger.info("Evolution completed")
            logger.info(f"Best strategy found: {self.best_strategy.grammar_string}")
            logger.info(f"Best strategy fitness: {self.best_strategy.fitness:.4f}")
            
            return self.best_strategy
            
        except Exception as e:
            logger.error(f"Error in strategy evolution: {str(e)}")
            return None
    
    def backtest_strategy(self, strategy=None, early_termination=True):
        """
        Backtest a strategy (either evolved or provided)
        
        Args:
            strategy (Strategy, optional): Strategy to test. If None, uses best_strategy
            early_termination (bool): Whether to terminate early if strategy performs poorly
            
        Returns:
            dict: Backtesting results
        """
        if strategy is None and self.best_strategy is None:
            logger.error("No strategy available for backtesting")
            return None
        
        strategy_to_test = strategy or self.best_strategy
        
        try:
            # Parse strategy using grammar parser
            if not strategy_to_test.parse(self.grammar_parser):
                logger.error("Failed to parse strategy")
                return None
            
            parsed_strategy = strategy_to_test.parsed_strategy
            if not parsed_strategy or 'entry_conditions' not in parsed_strategy or 'exit_conditions' not in parsed_strategy:
                logger.error("Invalid strategy format: missing entry or exit conditions")
                return None
            
            # Initialize results
            new_columns = {
                'Signal': 0,
                'Position': 0,
                'Strategy_Returns': 0
            }
            self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
            
            # Clear existing trades
            strategy_to_test.trades = []
            strategy_to_test.trade_history = []
            
            # Track current position
            current_position = 0
            entry_price = None
            entry_date = None
            
            # Apply entry and exit conditions
            for i in range(len(self.data)):
                date = self.data.index[i]
                price = self.data['Close'].iloc[i]
                
                # Check entry conditions
                if current_position <= 0:
                    try:
                        entry_signal = self.evaluate_conditions(parsed_strategy['entry_conditions'], i)
                        if entry_signal:
                            self.data.iloc[i, self.data.columns.get_loc('Signal')] = 1
                            current_position = 1
                            entry_price = price
                            entry_date = date
                            
                            # Record entry trade
                            strategy_to_test.add_trade({
                                'type': 'entry',
                                'date': date,
                                'price': price,
                                'position': current_position
                            })
                    except Exception as e:
                        logger.error(f"Error evaluating entry conditions: {e}")
                        continue
                
                # Check exit conditions
                elif current_position > 0:
                    try:
                        exit_signal = self.evaluate_conditions(parsed_strategy['exit_conditions'], i)
                        if exit_signal:
                            self.data.iloc[i, self.data.columns.get_loc('Signal')] = -1
                            
                            # Record exit trade
                            strategy_to_test.add_trade({
                                'type': 'exit',
                                'date': date,
                                'price': price,
                                'position': current_position,
                                'entry_price': entry_price,
                                'entry_date': entry_date,
                                'return': (price - entry_price) / entry_price
                            })
                            
                            current_position = 0
                            entry_price = None
                            entry_date = None
                    except Exception as e:
                        logger.error(f"Error evaluating exit conditions: {e}")
                        continue
            
            # Calculate positions and returns
            self.data['Position'] = self.data['Signal'].cumsum()
            self.data['Strategy_Returns'] = self.data['Position'].shift(1) * self.data['Close'].pct_change()
            
            # Calculate cumulative returns
            self.data['Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()
            
            # Early termination check
            if early_termination and len(self.data) > 20:
                recent_returns = self.data['Strategy_Returns'].tail(20)
                if recent_returns.mean() < -0.02:  # Stop if average daily return is less than -2%
                    logger.info("Early termination due to poor performance")
                    return None
            
            return self.analyze_performance()
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            return None
    
    def evaluate_conditions(self, conditions, index):
        """
        Evaluate a set of conditions at a specific index.
        
        Args:
            conditions (dict): Conditions to evaluate
            index (int): Data index to evaluate at
            
        Returns:
            bool: Whether conditions are met
        """
        try:
            if isinstance(conditions, dict):
                if 'operator' in conditions and 'conditions' in conditions:
                    # Compound condition
                    results = [self.evaluate_conditions(cond, index) for cond in conditions['conditions']]
                    if conditions['operator'] == 'AND':
                        return all(results)
                    elif conditions['operator'] == 'OR':
                        return any(results)
                    else:
                        logger.error(f"Unsupported operator: {conditions['operator']}")
                        return False
                elif conditions.get('type') == 'CONDITION':
                    # Single condition
                    return self.evaluate_single_condition(conditions, index)
                else:
                    logger.error("Invalid condition format: missing operator and conditions or type")
                    return False
            elif isinstance(conditions, list):
                # If conditions is a list, evaluate each condition and combine with AND
                return all(self.evaluate_conditions(cond, index) for cond in conditions)
                
            logger.error(f"Invalid conditions format: {type(conditions)}")
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating conditions: {str(e)}")
            return False
    
    def evaluate_single_condition(self, condition, index):
        """
        Evaluate a single condition at a specific index.
        
        Args:
            condition (dict): Single condition to evaluate
            index (int): Data index to evaluate at
            
        Returns:
            bool: Whether the condition is met
        """
        try:
            if condition.get('type') != 'CONDITION':
                logger.error(f"Invalid condition type: {condition.get('type')}")
                return False
            
            indicator = condition['indicator']
            operator = condition['operator']
            params = condition['parameters']
            reference = condition['reference']
            
            # Get current data point
            current_data = self.data.iloc[index]
            
            if indicator == 'RSI':
                period = int(params[0]) if params else 14
                threshold = float(reference)
                
                rsi_value = self.data[f"RSI_{period}"].iloc[index]
                
                if operator == '>':
                    return rsi_value > threshold
                elif operator == '<':
                    return rsi_value < threshold
                elif operator == 'CROSSES ABOVE':
                    if index > 0:
                        prev_rsi = self.data[f"RSI_{period}"].iloc[index-1]
                        return prev_rsi <= threshold and rsi_value > threshold
                    return False
                elif operator == 'CROSSES BELOW':
                    if index > 0:
                        prev_rsi = self.data[f"RSI_{period}"].iloc[index-1]
                        return prev_rsi >= threshold and rsi_value < threshold
                    return False
                else:
                    logger.error(f"Unknown operator for RSI: {operator}")
                    return False
                    
            elif indicator == 'MACD':
                fast_period = int(params[0]) if len(params) > 0 else 12
                slow_period = int(params[1]) if len(params) > 1 else 26
                signal_period = int(params[2]) if len(params) > 2 else 9
                
                macd_key = f"MACD_{fast_period}_{slow_period}_{signal_period}"
                signal_key = f"MACD_SIGNAL_{fast_period}_{slow_period}_{signal_period}"
                
                if macd_key not in self.data.columns or signal_key not in self.data.columns:
                    logger.error(f"MACD indicators not found: {macd_key}, {signal_key}")
                    return False
                
                macd_value = self.data[macd_key].iloc[index]
                signal_value = self.data[signal_key].iloc[index]
                
                if operator == 'CROSSES ABOVE':
                    if index > 0:
                        prev_macd = self.data[macd_key].iloc[index-1]
                        prev_signal = self.data[signal_key].iloc[index-1]
                        return prev_macd <= prev_signal and macd_value > signal_value
                    return False
                elif operator == 'CROSSES BELOW':
                    if index > 0:
                        prev_macd = self.data[macd_key].iloc[index-1]
                        prev_signal = self.data[signal_key].iloc[index-1]
                        return prev_macd >= prev_signal and macd_value < signal_value
                    return False
                else:
                    logger.error(f"Unknown operator for MACD: {operator}")
                    return False
                    
            elif indicator == 'BOLLINGER_BANDS':
                window = int(params[0]) if len(params) > 0 else 20
                std = float(params[1]) if len(params) > 1 else 2.0
                
                bb_key_upper = f"BB_upper_{window}_{std}"
                bb_key_lower = f"BB_lower_{window}_{std}"
                
                if bb_key_upper not in self.data.columns or bb_key_lower not in self.data.columns:
                    logger.error(f"Bollinger Bands indicators not found: {bb_key_upper}, {bb_key_lower}")
                    return False
                
                price = self.data['Close'].iloc[index]
                
                # Extract the band type from the reference
                if reference.startswith('BB_upper_'):
                    band = self.data[bb_key_upper].iloc[index]
                elif reference.startswith('BB_lower_'):
                    band = self.data[bb_key_lower].iloc[index]
                else:
                    logger.error(f"Invalid Bollinger Band reference: {reference}")
                    return False
                    
                if operator == 'CROSSES ABOVE':
                    if index > 0:
                        prev_price = self.data['Close'].iloc[index-1]
                        prev_band = self.data[reference].iloc[index-1]
                        return prev_price <= prev_band and price > band
                    return False
                elif operator == 'CROSSES BELOW':
                    if index > 0:
                        prev_price = self.data['Close'].iloc[index-1]
                        prev_band = self.data[reference].iloc[index-1]
                        return prev_price >= prev_band and price < band
                    return False
                elif operator == '>':
                    return price > band
                elif operator == '<':
                    return price < band
                else:
                    logger.error(f"Unknown operator for Bollinger Bands: {operator}")
                    return False
            
            else:
                logger.error(f"Unknown indicator: {indicator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating single condition: {str(e)}")
            return False
    
    def analyze_performance(self):
        """Analyze the performance of the best strategy"""
        if self.best_strategy is None:
            logger.error("No strategy available for analysis")
            return None
        
        try:
            # Get trade history
            trades = self.best_strategy.trade_history
            
            if not trades:
                logger.warning("No trades were executed")
                return {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'num_trades': 0,
                    'avg_trade_return': 0.0,
                    'profit_factor': 0.0
                }
            
            # Calculate performance metrics
            total_return = 0.0
            winning_trades = 0
            losing_trades = 0
            total_profit = 0.0
            total_loss = 0.0
            trade_returns = []
            
            # Process each trade
            for trade in trades:
                if trade['type'] == 'exit':
                    trade_return = trade['return']
                    trade_returns.append(trade_return)
                    
                    if trade_return > 0:
                        winning_trades += 1
                        total_profit += trade_return
                    else:
                        losing_trades += 1
                        total_loss += abs(trade_return)
            
            # Calculate metrics
            num_trades = len(trade_returns)
            win_rate = winning_trades / num_trades if num_trades > 0 else 0.0
            avg_trade_return = sum(trade_returns) / num_trades if num_trades > 0 else 0.0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate total return
            total_return = (1 + sum(trade_returns)) - 1
            
            # Calculate annual return
            start_date = trades[0]['date']
            end_date = trades[-1]['date']
            days = (end_date - start_date).days
            annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
            
            # Calculate Sharpe ratio
            returns_series = pd.Series(trade_returns)
            sharpe_ratio = np.sqrt(252) * returns_series.mean() / returns_series.std() if len(returns_series) > 1 else 0.0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns_series).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() if not drawdowns.empty else 0.0
            
            # Update strategy metrics
            self.best_strategy.metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': num_trades,
                'avg_trade_return': avg_trade_return,
                'profit_factor': profit_factor
            }
            
            # Log results
            logger.info(f"Performance Analysis Results:")
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Annual Return: {annual_return:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {max_drawdown:.2%}")
            logger.info(f"Win Rate: {win_rate:.2%}")
            logger.info(f"Number of Trades: {num_trades}")
            logger.info(f"Average Trade Return: {avg_trade_return:.2%}")
            logger.info(f"Profit Factor: {profit_factor:.2f}")
            
            return self.best_strategy.metrics
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return None
    
    def plot_results(self):
        """Plot analysis results"""
        if self.data is None or 'Signal' not in self.data.columns:
            logger.error("No results to plot")
            return
        
        try:
            # Set style
            plt.style.use('default')
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.grid': True,
                'grid.color': '#dddddd',
                'grid.linestyle': '--',
                'grid.alpha': 0.3,
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.figsize': (20, 15)
            })
            
            # Create a figure with a 2x2 grid for main plots
            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            
            # 1. Price and Signals Plot
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.data.index, self.data['Close'], label='Price', color='#1f77b4', alpha=0.7)
            
            # Plot buy signals
            buy_signals = self.data[self.data['Signal'] == 1]
            ax1.scatter(buy_signals.index, buy_signals['Close'],
                       marker='^', color='#2ecc71', s=100, label='Buy Signal')
            
            # Plot sell signals
            sell_signals = self.data[self.data['Signal'] == -1]
            ax1.scatter(sell_signals.index, sell_signals['Close'],
                       marker='v', color='#e74c3c', s=100, label='Sell Signal')
            
            ax1.set_title('Apple Stock Price with Trading Signals', pad=20)
            ax1.grid(True)
            ax1.legend(loc='upper left')
            
            # 2. Strategy Position Plot
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(self.data.index, self.data['Position'], 
                    label='Position', color='#9b59b6', linewidth=2)
            ax2.fill_between(self.data.index, self.data['Position'], 
                           color='#9b59b6', alpha=0.2)
            ax2.set_title('Strategy Position Over Time', pad=20)
            ax2.grid(True)
            ax2.legend(loc='upper left')
            
            # 3. Cumulative Returns Plot
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(self.data.index, self.data['Cumulative_Returns'], 
                    label='Strategy Returns', color='#2ecc71', linewidth=2)
            ax3.plot(self.data.index, self.data['Close'] / self.data['Close'].iloc[0], 
                    label='Buy & Hold', color='#95a5a6', alpha=0.5)
            ax3.set_title('Cumulative Strategy Returns vs Buy & Hold', pad=20)
            ax3.grid(True)
            ax3.legend(loc='upper left')
            
            # 4. Performance Metrics Table
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Get performance metrics
            metrics = self.analyze_performance()
            if metrics:
                # Create table data
                table_data = [
                    ['Total Return', f"{metrics['total_return']:.2%}"],
                    ['Annual Return', f"{metrics['annual_return']:.2%}"],
                    ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
                    ['Max Drawdown', f"{metrics['max_drawdown']:.2%}"],
                    ['Win Rate', f"{metrics['win_rate']:.2%}"],
                    ['Number of Trades', f"{metrics['num_trades']}"],
                    ['Avg Trade Return', f"{metrics['avg_trade_return']:.2%}"],
                    ['Profit Factor', f"{metrics['profit_factor']:.2f}"]
                ]
                
                # Create table
                table = ax4.table(cellText=table_data,
                                colLabels=['Metric', 'Value'],
                                cellLoc='center',
                                loc='center',
                                colWidths=[0.4, 0.4])
                
                # Style table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Set title
                ax4.set_title('Performance Metrics Summary', pad=20)
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Ensure plots directory exists
            if not os.path.exists('plots'):
                os.makedirs('plots')
            
            # Save the plot with a timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f'plots/trading_results_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trading results plot to: {plot_path}")
            
            # Display the plot
            plt.show()
            plt.close()
            
            # Generate pattern analysis plot
            pattern_analysis = self.analyze_strategy_patterns()
            if pattern_analysis:
                self._plot_pattern_analysis(pattern_analysis)
            
        except Exception as e:
            logger.error(f"Error in plotting: {str(e)}")
            raise  # Re-raise the exception to see the full traceback
    
    def analyze_strategy_patterns(self):
        """Analyze patterns and insights learned from the evolved strategies"""
        if self.best_strategy is None:
            logger.error("No strategy available for pattern analysis")
            return None
        
        try:
            # Get trade history
            trades = self.best_strategy.trade_history
            
            if not trades:
                logger.warning("No trades were executed for pattern analysis")
                return None
            
            # Initialize analysis results
            analysis = {
                'market_conditions': {},
                'indicator_patterns': {},
                'trade_timing': {},
                'risk_management': {}
            }
            
            # Analyze market conditions during trades
            for trade in trades:
                if trade['type'] == 'entry':
                    # Get market conditions at entry
                    entry_date = trade['date']
                    entry_data = self.data.loc[entry_date]
                    
                    # Analyze RSI conditions
                    for period in [7, 9, 14, 21, 30]:
                        rsi_key = f'RSI_{period}'
                        if rsi_key not in analysis['market_conditions']:
                            analysis['market_conditions'][rsi_key] = []
                        analysis['market_conditions'][rsi_key].append(entry_data[rsi_key])
                    
                    # Analyze Bollinger Bands
                    for window in [10, 15, 20, 30]:
                        for std in [1.5, 2.0, 2.5, 3.0]:
                            bb_key = f'BB_middle_{window}_{std}'
                            if bb_key not in analysis['market_conditions']:
                                analysis['market_conditions'][bb_key] = []
                            analysis['market_conditions'][bb_key].append(entry_data[bb_key])
            
            # Calculate statistics for each indicator
            for indicator, values in analysis['market_conditions'].items():
                if values:
                    analysis['indicator_patterns'][indicator] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values),
                        'median': np.median(values)
                    }
            
            # Analyze trade timing
            trade_durations = []
            for i in range(0, len(trades)-1, 2):
                if trades[i]['type'] == 'entry' and trades[i+1]['type'] == 'exit':
                    duration = (trades[i+1]['date'] - trades[i]['date']).days
                    trade_durations.append(duration)
            
            if trade_durations:
                analysis['trade_timing'] = {
                    'avg_duration': np.mean(trade_durations),
                    'min_duration': min(trade_durations),
                    'max_duration': max(trade_durations),
                    'median_duration': np.median(trade_durations)
                }
            
            # Analyze risk management patterns
            winning_trades = []
            losing_trades = []
            for trade in trades:
                if trade['type'] == 'exit':
                    if trade['return'] > 0:
                        winning_trades.append(trade['return'])
                    else:
                        losing_trades.append(trade['return'])
            
            analysis['risk_management'] = {
                'avg_win': np.mean(winning_trades) if winning_trades else 0,
                'avg_loss': np.mean(losing_trades) if losing_trades else 0,
                'win_loss_ratio': len(winning_trades) / len(losing_trades) if losing_trades else float('inf'),
                'max_consecutive_wins': self._calculate_max_consecutive(winning_trades),
                'max_consecutive_losses': self._calculate_max_consecutive(losing_trades)
            }
            
            # Create visualizations
            self._plot_pattern_analysis(analysis)
            
            # Log concise results
            logger.info("\n📊 Strategy Pattern Analysis Summary:")
            logger.info("----------------------------------------")
            
            # Market Conditions Summary
            logger.info("\n📈 Market Conditions:")
            best_rsi = max(analysis['indicator_patterns'].items(), 
                          key=lambda x: x[1]['mean'] if 'RSI' in x[0] else -float('inf'))
            best_bb = max(analysis['indicator_patterns'].items(), 
                         key=lambda x: x[1]['mean'] if 'BB' in x[0] else -float('inf'))
            
            logger.info(f"Most Effective Indicators:")
            logger.info(f"  • {best_rsi[0]}: {best_rsi[1]['mean']:.1f} (Range: {best_rsi[1]['min']:.1f}-{best_rsi[1]['max']:.1f})")
            logger.info(f"  • {best_bb[0]}: {best_bb[1]['mean']:.1f} (Range: {best_bb[1]['min']:.1f}-{best_bb[1]['max']:.1f})")
            
            # Trade Timing Summary
            logger.info("\n⏱️ Trade Timing:")
            timing = analysis['trade_timing']
            logger.info(f"  • Average Duration: {timing['avg_duration']:.1f} days")
            logger.info(f"  • Range: {timing['min_duration']:.0f}-{timing['max_duration']:.0f} days")
            
            # Risk Management Summary
            logger.info("\n🎯 Risk Management:")
            risk = analysis['risk_management']
            logger.info(f"  • Win Rate: {len(winning_trades)/len(trades)*100:.1f}%")
            logger.info(f"  • Avg Win: {risk['avg_win']*100:.1f}% | Avg Loss: {risk['avg_loss']*100:.1f}%")
            logger.info(f"  • Max Streak: {risk['max_consecutive_wins']} wins | {risk['max_consecutive_losses']} losses")
            
            logger.info("\n💡 Key Insights:")
            logger.info("----------------------------------------")
            self._generate_insights(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return None
    
    def _plot_pattern_analysis(self, analysis):
        """Create visualizations for pattern analysis"""
        try:
            # Set style
            plt.style.use('default')
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.grid': True,
                'grid.color': '#dddddd',
                'grid.linestyle': '--',
                'grid.alpha': 0.3,
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.figsize': (15, 10)
            })
            
            # Create figure with subplots
            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            
            # 1. Indicator Distribution
            ax1 = fig.add_subplot(gs[0, 0])
            indicators = [k for k in analysis['indicator_patterns'].keys() if 'RSI' in k]
            values = [analysis['indicator_patterns'][k]['mean'] for k in indicators]
            ax1.bar(indicators, values, color='#3498db')
            ax1.set_title('RSI Values at Entry Points')
            ax1.set_ylabel('Average Value')
            plt.xticks(rotation=45)
            
            # 2. Trade Duration Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            durations = [t['duration'] for t in self.best_strategy.trade_history if 'duration' in t]
            ax2.hist(durations, bins=20, color='#2ecc71', edgecolor='black')
            ax2.set_title('Trade Duration Distribution')
            ax2.set_xlabel('Days')
            ax2.set_ylabel('Number of Trades')
            
            # 3. Win/Loss Distribution
            ax3 = fig.add_subplot(gs[1, 0])
            returns = [t['return'] for t in self.best_strategy.trade_history if t['type'] == 'exit']
            ax3.hist(returns, bins=20, color='#e74c3c', edgecolor='black')
            ax3.set_title('Trade Return Distribution')
            ax3.set_xlabel('Return (%)')
            ax3.set_ylabel('Number of Trades')
            
            # 4. Risk Metrics
            ax4 = fig.add_subplot(gs[1, 1])
            metrics = ['Win Rate', 'Avg Win', 'Avg Loss', 'Max Streak']
            values = [
                len([t for t in self.best_strategy.trade_history if t['type'] == 'exit' and t['return'] > 0]) / len([t for t in self.best_strategy.trade_history if t['type'] == 'exit']),
                analysis['risk_management']['avg_win'],
                analysis['risk_management']['avg_loss'],
                analysis['risk_management']['max_consecutive_wins']
            ]
            ax4.bar(metrics, values, color='#9b59b6')
            ax4.set_title('Risk Management Metrics')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save the plot with a timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f'plots/pattern_analysis_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pattern analysis plot to: {plot_path}")
            
            # Display the plot
            plt.show()
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating pattern analysis plots: {str(e)}")
            raise  # Re-raise the exception to see the full traceback
    
    def _generate_insights(self, analysis):
        """Generate key insights from the analysis"""
        try:
            # Market Condition Insights
            rsi_patterns = {k: v for k, v in analysis['indicator_patterns'].items() if 'RSI' in k}
            best_rsi = max(rsi_patterns.items(), key=lambda x: x[1]['mean'])
            
            logger.info(f"1. The strategy performs best when {best_rsi[0]} is around {best_rsi[1]['mean']:.1f}")
            
            # Trade Timing Insights
            timing = analysis['trade_timing']
            if timing['avg_duration'] < 5:
                logger.info("2. The strategy prefers short-term trades (typically < 5 days)")
            elif timing['avg_duration'] < 20:
                logger.info("2. The strategy prefers medium-term trades (5-20 days)")
            else:
                logger.info("2. The strategy prefers long-term trades (> 20 days)")
            
            # Risk Management Insights
            risk = analysis['risk_management']
            if risk['win_loss_ratio'] > 2:
                logger.info("3. The strategy has a strong win/loss ratio, indicating good risk management")
            elif risk['win_loss_ratio'] > 1:
                logger.info("3. The strategy has a positive win/loss ratio, but could improve risk management")
            else:
                logger.info("3. The strategy needs improvement in risk management")
            
            # Overall Performance Insight
            if risk['max_consecutive_wins'] > risk['max_consecutive_losses']:
                logger.info("4. The strategy shows consistency with longer winning streaks than losing streaks")
            else:
                logger.info("4. The strategy needs to improve consistency in trade outcomes")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
    
    def _calculate_max_consecutive(self, returns):
        """Calculate maximum consecutive wins or losses"""
        if not returns:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for ret in returns:
            if ret > 0:  # For wins
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak

def main():
    # Initialize the bot
    bot = EvolvedAppleBot(start_date='2022-01-01')
    
    # Fetch and prepare data
    if bot.fetch_data() is None:
        logger.error("Failed to load data. Exiting...")
        return
    
    # Calculate indicators
    bot.calculate_indicators()
    
    # Evolve strategies
    logger.info("Starting strategy evolution...")
    best_strategy = bot.evolve_strategies(population_size=50, generations=20)
    
    if best_strategy is None:
        logger.error("Strategy evolution failed. Exiting...")
        return
    
    # Backtest the best strategy
    logger.info("Backtesting best strategy...")
    if bot.backtest_strategy() is None:
        logger.error("Backtesting failed. Exiting...")
        return
    
    # Analyze and display results
    bot.analyze_performance()
    bot.plot_results()

if __name__ == "__main__":
    main() 