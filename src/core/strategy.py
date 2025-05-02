import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Strategy')

class Strategy:
    """
    Represents a trading strategy in the evolution process.
    
    Attributes:
        grammar_string (str): The strategy expressed in grammar format
        fitness (float): The fitness score of the strategy
        generation (int): The generation in which this strategy was created
        trades (list): List of trades executed by this strategy
        metrics (dict): Performance metrics of the strategy
    """
    
    def __init__(self, grammar_string=None, generation=0):
        self.grammar_string = grammar_string
        self.fitness = 0.0
        self.generation = generation
        self.trades = []  # List to store trade history
        self.trade_history = []  # Alias for trades to maintain compatibility
        self.initial_portfolio = 0.0
        self.final_portfolio = 0.0
        self.profit_percentage = 0.0
        self.improvement = 0.0
        self.parsed_strategy = None
        self.metrics = {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'num_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    def add_trade(self, trade):
        """
        Add a trade to the trade history.
        
        Args:
            trade (dict): Trade information including type, price, date, etc.
        """
        self.trades.append(trade)
        self.trade_history = self.trades  # Keep trade_history in sync
        self.metrics['num_trades'] = len(self.trades)
        logger.info(f"Added trade: {trade}")
    
    def update_fitness(self, metrics):
        """
        Update the strategy's fitness based on performance metrics.
        
        The fitness function combines multiple metrics:
        - Total return (40% weight)
        - Sharpe ratio (30% weight)
        - Win rate (20% weight)
        - Inverse of max drawdown (10% weight)
        
        Args:
            metrics (dict): Performance metrics dictionary
        """
        self.metrics = metrics
        
        # Normalize metrics to a 0-1 scale
        norm_return = max(0, min(1, (metrics['total_return'] + 100) / 200))  # Assuming returns between -100% and +100%
        norm_sharpe = max(0, min(1, (metrics['sharpe_ratio'] + 3) / 6))  # Assuming Sharpe between -3 and 3
        norm_winrate = metrics['win_rate'] / 100  # Win rate is already 0-100
        norm_drawdown = 1 - abs(metrics['max_drawdown'] / 100)  # Convert drawdown to positive score
        
        # Calculate weighted fitness
        self.fitness = (
            0.4 * norm_return +
            0.3 * norm_sharpe +
            0.2 * norm_winrate +
            0.1 * norm_drawdown
        )
    
    def __str__(self):
        """String representation of the strategy"""
        return (f"Strategy(gen={self.generation}, fitness={self.fitness:.4f}, "
                f"return={self.metrics['total_return']:.2f}%, "
                f"sharpe={self.metrics['sharpe_ratio']:.2f})")
    
    def __repr__(self):
        """Detailed representation of the strategy"""
        return (f"Strategy(\n"
                f"  grammar='{self.grammar_string}'\n"
                f"  generation={self.generation}\n"
                f"  fitness={self.fitness:.4f}\n"
                f"  metrics={self.metrics}\n"
                f")")
    
    def copy(self):
        """Create a copy of the strategy"""
        new_strategy = Strategy(self.grammar_string, self.generation)
        new_strategy.fitness = self.fitness
        new_strategy.metrics = self.metrics.copy()
        new_strategy.trades = self.trades.copy()
        return new_strategy
    
    def split(self):
        """
        Split the strategy's grammar string into entry and exit conditions.
        
        Returns:
            tuple: (entry_conditions, exit_conditions)
        """
        try:
            parts = self.grammar_string.split(" SELL WHEN ")
            if len(parts) != 2:
                logger.error("Invalid strategy format for splitting")
                return None, None
                
            entry_part = parts[0].replace("BUY WHEN ", "").strip()
            exit_part = parts[1].strip()
            
            return entry_part, exit_part
            
        except Exception as e:
            logger.error(f"Error splitting strategy: {e}")
            return None, None
    
    def parse(self, parser):
        """
        Parse the strategy's grammar string.
        
        Args:
            parser: GrammarParser instance
            
        Returns:
            bool: True if parsing successful, False otherwise
        """
        try:
            self.parsed_strategy = parser.parse(self.grammar_string)
            logger.info(f"Successfully parsed strategy: {self.grammar_string[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to parse strategy: {e}")
            return False
    
    def calculate_metrics(self, portfolio_history):
        """
        Calculate performance metrics from trade history and portfolio values.
        
        Args:
            portfolio_history (pd.Series): Historical portfolio values
            
        Returns:
            dict: Dictionary of updated metrics
        """
        if len(portfolio_history) < 2:
            logger.warning("Insufficient data to calculate metrics")
            return self.metrics
        
        # Calculate daily returns
        daily_returns = portfolio_history.pct_change().dropna()
        
        # Initial and final portfolio values
        self.initial_portfolio = portfolio_history.iloc[0]
        self.final_portfolio = portfolio_history.iloc[-1]
        
        # Calculate profit percentage
        self.profit_percentage = ((self.final_portfolio / self.initial_portfolio) - 1) * 100
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        if len(daily_returns) > 0 and daily_returns.std() != 0:
            self.metrics['sharpe_ratio'] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        # Calculate Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        self.metrics['max_drawdown'] = abs(drawdown.min()) * 100  # Convert to percentage
        
        # Calculate win rate and other trade-specific metrics if trades are available
        if self.trades:
            # Count actual trades (only SELL trades have profit values)
            sell_trades = [t for t in self.trades if t.get('type') == 'SELL']
            winning_trades = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
            total_sell_trades = len(sell_trades)
            
            self.metrics['win_rate'] = (winning_trades / total_sell_trades) * 100 if total_sell_trades > 0 else 0
            
            # Calculate average win and loss
            wins = [t for t in sell_trades if t.get('profit', 0) > 0]
            losses = [t for t in sell_trades if t.get('profit', 0) <= 0]
            
            if len(wins) > 0:
                self.metrics['avg_win'] = sum(t.get('profit', 0) for t in wins) / len(wins)
            
            if len(losses) > 0:
                self.metrics['avg_loss'] = sum(t.get('profit', 0) for t in losses) / len(losses)
            
            # Calculate profit factor
            total_wins = sum(t.get('profit', 0) for t in wins)
            total_losses = abs(sum(t.get('profit', 0) for t in losses)) if losses else 1
            
            self.metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else total_wins
        
        logger.info(f"Metrics calculated for strategy. Profit: {self.profit_percentage:.2f}%, "
                   f"Sharpe: {self.metrics['sharpe_ratio']:.2f}, "
                   f"Max Drawdown: {self.metrics['max_drawdown']:.2f}%")
        
        return self.metrics
    
    def compare_to(self, other_strategy):
        """
        Compare performance to another strategy.
        
        Args:
            other_strategy (Strategy): Strategy to compare with
            
        Returns:
            bool: True if this strategy is better, False otherwise
        """
        # Primary comparison based on profit percentage
        if self.profit_percentage != other_strategy.profit_percentage:
            return self.profit_percentage > other_strategy.profit_percentage
        
        # If profit percentage is the same, use Sharpe ratio as tiebreaker
        if self.metrics['sharpe_ratio'] != other_strategy.metrics['sharpe_ratio']:
            return self.metrics['sharpe_ratio'] > other_strategy.metrics['sharpe_ratio']
        
        # If Sharpe is also the same, use maximum drawdown (lower is better)
        return self.metrics['max_drawdown'] < other_strategy.metrics['max_drawdown']
    
    def to_dict(self):
        """
        Convert strategy to dictionary for serialization.
        
        Returns:
            dict: Strategy attributes in dictionary form
        """
        # Deep copy trades to handle pandas.Timestamp objects
        trades_copy = []
        for trade in self.trades:
            trade_copy = dict(trade)
            # Convert Timestamp objects to string if present
            if 'date' in trade_copy and hasattr(trade_copy['date'], 'strftime'):
                trade_copy['date'] = trade_copy['date'].strftime('%Y-%m-%d')
            trades_copy.append(trade_copy)
        
        return {
            'grammar_string': self.grammar_string,
            'initial_portfolio': self.initial_portfolio,
            'final_portfolio': self.final_portfolio,
            'profit_percentage': self.profit_percentage,
            'generation': self.generation,
            'improvement': self.improvement,
            'trades': trades_copy,
            'metrics': self.metrics
        }
    
    def to_json(self):
        """
        Convert strategy to JSON string.
        
        Returns:
            str: JSON representation of the strategy
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a strategy from a dictionary.
        
        Args:
            data (dict): Dictionary with strategy attributes
            
        Returns:
            Strategy: New strategy instance
        """
        strategy = cls(data['grammar_string'], data['generation'])
        strategy.initial_portfolio = data['initial_portfolio']
        strategy.final_portfolio = data['final_portfolio']
        strategy.profit_percentage = data['profit_percentage']
        strategy.improvement = data['improvement']
        strategy.trades = data['trades']
        strategy.metrics = data['metrics']
        return strategy
    
    @classmethod
    def from_json(cls, json_str):
        """
        Create a strategy from a JSON string.
        
        Args:
            json_str (str): JSON representation of a strategy
            
        Returns:
            Strategy: New strategy instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data) 