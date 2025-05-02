import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from typing import Dict, Any, Optional, List, Union, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Backtesting')

class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    
    This class is responsible for:
    - Generating trading signals from strategy conditions
    - Simulating portfolio performance
    - Tracking trade history and positions
    - Calculating performance metrics
    - Visualizing equity curves and trade history
    """
    
    def __init__(self, historical_data, indicator_calculator, params=None):
        """
        Initialize the backtesting engine.
        
        Args:
            historical_data (pd.DataFrame): OHLCV data for backtesting
            indicator_calculator: IndicatorCalculator instance 
            params (dict, optional): Backtesting parameters
        """
        self.historical_data = historical_data
        self.indicator_calculator = indicator_calculator
        
        # Default parameters
        self.params = {
            'initial_capital': 10000,
            'transaction_cost': 0.001,  # 0.1% per trade
            'slippage': 0.001,  # 0.1% slippage assumption
            'risk_per_trade': 0.02,  # 2% of portfolio per trade
            'enable_fractional_shares': True,
            'entry_time': 'close',  # 'open' or 'close'
            'exit_time': 'close',  # 'open' or 'close'
            'early_termination_threshold': -15.0,  # Terminate if drawdown exceeds this percentage
            'early_termination_period': 20,  # Check performance after this many days
            'performance_sample_size': 100,  # Sample size for estimating full performance
        }
        
        # Override with user parameters if provided
        if params:
            self.params.update(params)
        
        logger.info("BacktestEngine initialized with"
                   f" {len(historical_data)} days of data and"
                   f" ${self.params['initial_capital']} initial capital")
        
        # Pre-allocate memory for portfolio history
        self._initialize_memory_buffers()
    
    def _initialize_memory_buffers(self):
        """Initialize memory buffers for improved performance"""
        # Pre-allocate memory for commonly used arrays
        self.portfolio_values = np.zeros(len(self.historical_data))
        self.cash_values = np.zeros(len(self.historical_data))
        self.position_values = np.zeros(len(self.historical_data))
    
    def run_backtest(self, strategy, condition_evaluator=None, early_termination=True):
        """
        Run a backtest on a trading strategy.
        
        Args:
            strategy: Strategy object with parsed strategy components
            condition_evaluator: Optional function for evaluating conditions
            early_termination: If True, will terminate early for clearly poor strategies
            
        Returns:
            dict: Backtesting results
        """
        logger.info(f"Starting backtest for strategy: {strategy.grammar_string[:50]}...")
        
        # Ensure strategy is parsed
        if not strategy.parsed_strategy:
            logger.error("Strategy is not parsed, cannot backtest")
            return None
        
        # Initialize portfolio and trade log
        portfolio = self._initialize_portfolio()
        
        # Generate trading signals
        signals = self._generate_signals(strategy, condition_evaluator)
        logger.info(f"Generated {sum(signals['entry'])} entry signals and {sum(signals['exit'])} exit signals")
        
        # Run portfolio simulation with early termination if enabled
        try:
            if early_termination:
                portfolio = self._simulate_portfolio_with_early_termination(signals, portfolio)
            else:
                portfolio = self._simulate_portfolio(signals, portfolio)
        except EarlyTerminationException as e:
            # Handle early termination
            logger.info(f"Early termination triggered: {str(e)}")
            # Calculate metrics with partial data
            performance_metrics = self._calculate_metrics(portfolio)
            
            # Update strategy with trades and results
            strategy.trades = portfolio['trades']
            strategy.initial_portfolio = self.params['initial_capital']
            strategy.final_portfolio = portfolio['history'][-1]['portfolio_value'] if portfolio['history'] else strategy.initial_portfolio
            strategy.profit_percentage = performance_metrics['profit_percentage']
            strategy.metrics.update(performance_metrics)
            strategy.early_terminated = True
            
            # Return partial results
            portfolio_history_df = pd.DataFrame(portfolio['history'])
            if not portfolio_history_df.empty:
                portfolio_history_df.set_index('date', inplace=True)
            
            return {
                'portfolio_history': portfolio_history_df,
                'trades': portfolio['trades'],
                'metrics': performance_metrics,
                'early_terminated': True
            }
        
        # Calculate performance metrics for completed backtest
        performance_metrics = self._calculate_metrics(portfolio)
        
        # Update strategy with trades and results
        strategy.trades = portfolio['trades']
        strategy.initial_portfolio = self.params['initial_capital']
        strategy.final_portfolio = portfolio['history'][-1]['portfolio_value'] if portfolio['history'] else strategy.initial_portfolio
        strategy.profit_percentage = performance_metrics['profit_percentage']
        strategy.metrics.update(performance_metrics)
        strategy.early_terminated = False
        
        logger.info(f"Backtest completed: {len(portfolio['trades'])} trades, "
                   f"{performance_metrics['profit_percentage']:.2f}% profit")
        
        # Convert portfolio history to DataFrame for easier analysis
        portfolio_history_df = pd.DataFrame(portfolio['history'])
        if not portfolio_history_df.empty:
            portfolio_history_df.set_index('date', inplace=True)
        
        return {
            'portfolio_history': portfolio_history_df,
            'trades': portfolio['trades'],
            'metrics': performance_metrics,
            'early_terminated': False
        }
    
    def _initialize_portfolio(self):
        """
        Initialize a portfolio for backtesting.
        
        Returns:
            dict: Initial portfolio state
        """
        return {
            'cash': self.params['initial_capital'],
            'position': 0,
            'entry_price': 0,
            'trades': [],
            'history': []
        }
    
    def _generate_signals(self, strategy, condition_evaluator=None):
        """
        Generate trading signals from a strategy.
        
        Args:
            strategy: Strategy object with parsed strategy components
            condition_evaluator: Optional condition evaluation function
            
        Returns:
            pd.DataFrame: DataFrame with entry and exit signals
        """
        signals = pd.DataFrame(index=self.historical_data.index)
        signals['entry'] = False
        signals['exit'] = False
        
        # Use default condition evaluator if none provided
        if condition_evaluator is None:
            condition_evaluator = self._evaluate_condition
        
        # Extract conditions from strategy
        entry_conditions = strategy.parsed_strategy['entry_conditions']
        exit_conditions = strategy.parsed_strategy['exit_conditions']
        
        # Generate signals for each day
        for i in range(len(self.historical_data)):
            # Skip if not enough data for indicators
            if i < 30:  # Minimum data points for indicators
                continue
            
            current_data = self.historical_data.iloc[:i+1]
            date = self.historical_data.index[i]
            
            # Check entry condition
            entry_condition = condition_evaluator(
                entry_conditions,
                current_data,
                i
            )
            
            # Check exit condition
            exit_condition = condition_evaluator(
                exit_conditions,
                current_data,
                i
            )
            
            # Set signals
            signals.loc[date, 'entry'] = entry_condition
            signals.loc[date, 'exit'] = exit_condition
        
        return signals
    
    def _simulate_portfolio_with_early_termination(self, signals, portfolio):
        """
        Simulate portfolio performance with early termination for poor strategies.
        
        Args:
            signals (pd.DataFrame): Trading signals
            portfolio (dict): Portfolio state
            
        Returns:
            dict: Updated portfolio with performance history
            
        Raises:
            EarlyTerminationException: If strategy is terminated early
        """
        in_position = False
        
        # Parameters for early termination
        early_term_threshold = self.params['early_termination_threshold']
        early_term_period = self.params['early_termination_period']
        check_point = early_term_period
        
        # For tracking drawdown for early termination
        peak_value = self.params['initial_capital']
        
        # Process each day
        for i, date in enumerate(signals.index):
            current_price = self.historical_data.loc[date]['Close']
            
            # Update portfolio value history
            portfolio_value = portfolio['cash']
            if in_position:
                portfolio_value += portfolio['position'] * current_price
            
            # Check for new peak
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            
            # Calculate current drawdown
            current_drawdown = ((portfolio_value / peak_value) - 1) * 100
            
            # Check for early termination
            if i >= check_point and current_drawdown <= early_term_threshold:
                # Calculate performance metrics so far
                if len(portfolio['history']) > 0:
                    relative_change = (portfolio_value / portfolio['history'][0]['portfolio_value'] - 1) * 100
                    logger.info(f"Terminating strategy early at day {i} with {current_drawdown:.2f}% drawdown, {relative_change:.2f}% change")
                else:
                    logger.info(f"Terminating strategy early at day {i} with {current_drawdown:.2f}% drawdown")
                
                raise EarlyTerminationException(f"Strategy exceeds drawdown threshold: {current_drawdown:.2f}%")
            
            # Record portfolio state
            portfolio['history'].append({
                'date': date,
                'cash': portfolio['cash'],
                'position': portfolio['position'],
                'portfolio_value': portfolio_value
            })
            
            # Skip if not enough data for trading
            if i < 30:
                continue
            
            # Check exit signal if in position
            if in_position and signals.loc[date]['exit']:
                # Apply slippage and transaction cost
                exit_price = current_price * (1 - self.params['slippage'])
                cash_gained = portfolio['position'] * exit_price
                transaction_fee = cash_gained * self.params['transaction_cost']
                
                # Calculate profit/loss
                profit = (exit_price - portfolio['entry_price']) * portfolio['position']
                profit_percentage = (exit_price / portfolio['entry_price'] - 1) * 100
                
                # Record trade
                trade = {
                    'type': 'SELL',
                    'date': date,
                    'price': exit_price,
                    'shares': portfolio['position'],
                    'value': cash_gained,
                    'fee': transaction_fee,
                    'profit': profit,
                    'profit_percentage': profit_percentage
                }
                portfolio['trades'].append(trade)
                
                # Update portfolio
                portfolio['cash'] += cash_gained - transaction_fee
                portfolio['position'] = 0
                portfolio['entry_price'] = 0
                in_position = False
                
                logger.debug(f"Sold position at {date}: {trade}")
            
            # Check entry signal if not in position
            elif not in_position and signals.loc[date]['entry']:
                # Apply slippage
                entry_price = current_price * (1 + self.params['slippage'])
                
                # Calculate position size based on risk per trade
                risk_amount = portfolio['cash'] * self.params['risk_per_trade']
                position_size = risk_amount / entry_price
                
                # Check if we have enough cash
                cash_required = position_size * entry_price
                transaction_fee = cash_required * self.params['transaction_cost']
                total_cost = cash_required + transaction_fee
                
                if total_cost <= portfolio['cash']:
                    # Record trade
                    trade = {
                        'type': 'BUY',
                        'date': date,
                        'price': entry_price,
                        'shares': position_size,
                        'value': cash_required,
                        'fee': transaction_fee
                    }
                    portfolio['trades'].append(trade)
                    
                    # Update portfolio
                    portfolio['cash'] -= total_cost
                    portfolio['position'] = position_size
                    portfolio['entry_price'] = entry_price
                    in_position = True
                    
                    logger.debug(f"Bought position at {date}: {trade}")
        
        return portfolio
    
    def _simulate_portfolio(self, signals, portfolio):
        """
        Simulate portfolio performance based on signals.
        
        Args:
            signals (pd.DataFrame): Trading signals
            portfolio (dict): Portfolio state
            
        Returns:
            dict: Updated portfolio with performance history
        """
        in_position = False
        
        # Process each day
        for i, date in enumerate(signals.index):
            current_price = self.historical_data.loc[date]['Close']
            
            # Update portfolio value history
            portfolio_value = portfolio['cash']
            if in_position:
                portfolio_value += portfolio['position'] * current_price
            
            # Record portfolio state
            portfolio['history'].append({
                'date': date,
                'cash': portfolio['cash'],
                'position': portfolio['position'],
                'portfolio_value': portfolio_value
            })
            
            # Skip if not enough data for trading
            if i < 30:
                continue
            
            # Check exit signal if in position
            if in_position and signals.loc[date]['exit']:
                # Apply slippage and transaction cost
                exit_price = current_price * (1 - self.params['slippage'])
                cash_gained = portfolio['position'] * exit_price
                transaction_fee = cash_gained * self.params['transaction_cost']
                
                # Calculate profit/loss
                profit = (exit_price - portfolio['entry_price']) * portfolio['position']
                profit_percentage = (exit_price / portfolio['entry_price'] - 1) * 100
                
                # Record trade
                trade = {
                    'type': 'SELL',
                    'date': date,
                    'price': exit_price,
                    'shares': portfolio['position'],
                    'value': cash_gained,
                    'fee': transaction_fee,
                    'profit': profit,
                    'profit_percentage': profit_percentage
                }
                portfolio['trades'].append(trade)
                
                # Update portfolio
                portfolio['cash'] += cash_gained - transaction_fee
                portfolio['position'] = 0
                portfolio['entry_price'] = 0
                in_position = False
                
                logger.debug(f"Sold position at {date}: {trade}")
            
            # Check entry signal if not in position
            elif not in_position and signals.loc[date]['entry']:
                # Apply slippage
                entry_price = current_price * (1 + self.params['slippage'])
                
                # Calculate position size based on risk per trade
                risk_amount = portfolio['cash'] * self.params['risk_per_trade']
                position_size = risk_amount / entry_price
                
                # Check if we have enough cash
                cash_required = position_size * entry_price
                transaction_fee = cash_required * self.params['transaction_cost']
                total_cost = cash_required + transaction_fee
                
                if total_cost <= portfolio['cash']:
                    # Record trade
                    trade = {
                        'type': 'BUY',
                        'date': date,
                        'price': entry_price,
                        'shares': position_size,
                        'value': cash_required,
                        'fee': transaction_fee
                    }
                    portfolio['trades'].append(trade)
                    
                    # Update portfolio
                    portfolio['cash'] -= total_cost
                    portfolio['position'] = position_size
                    portfolio['entry_price'] = entry_price
                    in_position = True
                    
                    logger.debug(f"Bought position at {date}: {trade}")
        
        return portfolio
    
    def _calculate_metrics(self, portfolio):
        """
        Calculate performance metrics from portfolio history.
        
        Args:
            portfolio (dict): Portfolio with history and trades
            
        Returns:
            dict: Performance metrics
        """
        if not portfolio['history']:
            return {
                'profit_percentage': -100,
                'sharpe_ratio': 0,
                'max_drawdown': 100,
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Convert portfolio history to pandas Series
        portfolio_values = pd.Series(
            [day['portfolio_value'] for day in portfolio['history']],
            index=[day['date'] for day in portfolio['history']]
        )
        
        # Create a temporary strategy to calculate metrics
        temp_strategy = Strategy()
        temp_strategy.trades = portfolio['trades']
        metrics = temp_strategy.calculate_metrics(portfolio_values)
        
        # Add additional metrics
        metrics['num_trades'] = len(portfolio['trades'])
        
        return metrics
    
    def _evaluate_condition(self, condition, data, idx):
        """
        Evaluate a condition for a specific data point.
        This is a placeholder and should be replaced with actual evaluation logic.
        
        Args:
            condition (dict): Condition from parsed strategy
            data (pd.DataFrame): Historical data up to the current point
            idx (int): Index of the current data point
            
        Returns:
            bool: Whether the condition is met
        """
        # Delegate condition evaluation to Bot._evaluate_condition
        # This is just a placeholder - in practice, the condition evaluation
        # logic would be implemented here or passed in as a function
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
            # Risk-based position sizing
            risk_percentage = position_sizing['risk_percentage'] / 100
            return (available_cash * risk_percentage) / price
        
        # Default to using risk per trade parameter
        return (available_cash * self.params['risk_per_trade']) / price
    
    def plot_equity_curve(self, portfolio_history, save_path=None):
        """
        Plot the equity curve for a backtest.
        
        Args:
            portfolio_history (pd.DataFrame): Portfolio history
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_history.index, portfolio_history['portfolio_value'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        
        return plt.gcf()
    
    def plot_trade_analysis(self, trades, portfolio_history, save_path=None):
        """
        Plot trade analysis metrics.
        
        Args:
            trades (list): List of trades
            portfolio_history (pd.DataFrame): Portfolio history
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure
        """
        # Filter for completed trades
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        if not sell_trades:
            logger.warning("No completed trades to analyze")
            return None
        
        # Extract trade metrics
        profits = [t['profit'] for t in sell_trades]
        dates = [t['date'] for t in sell_trades]
        profit_percentages = [t['profit_percentage'] for t in sell_trades]
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot trade profit
        axs[0].bar(range(len(profits)), profits, color=['g' if p > 0 else 'r' for p in profits])
        axs[0].set_title('Trade Profit/Loss')
        axs[0].set_xlabel('Trade Number')
        axs[0].set_ylabel('Profit/Loss ($)')
        axs[0].grid(True)
        
        # Plot cumulative profit
        cumulative_profits = np.cumsum(profits)
        axs[1].plot(range(len(cumulative_profits)), cumulative_profits)
        axs[1].set_title('Cumulative Profit/Loss')
        axs[1].set_xlabel('Trade Number')
        axs[1].set_ylabel('Cumulative Profit/Loss ($)')
        axs[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Trade analysis saved to {save_path}")
        
        return fig


class EarlyTerminationException(Exception):
    """Exception raised when a strategy is terminated early due to poor performance."""
    pass 