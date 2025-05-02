import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from strategy import Strategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('WalkForwardValidator')

class WalkForwardValidator:
    """
    Implements walk-forward optimization for strategy validation.
    
    The walk-forward optimization process:
    1. Split data into multiple in-sample and out-of-sample periods
    2. Optimize strategy on in-sample data
    3. Test strategy on out-of-sample data
    4. Calculate consistency metrics across all periods
    """
    
    def __init__(self,
                 in_sample_ratio: float = 0.7,
                 num_periods: int = 5,
                 min_period_length: int = 100):
        """
        Initialize walk-forward validator.
        
        Args:
            in_sample_ratio: Ratio of in-sample to total period length
            num_periods: Number of walk-forward periods
            min_period_length: Minimum number of data points per period
        """
        self.in_sample_ratio = in_sample_ratio
        self.num_periods = num_periods
        self.min_period_length = min_period_length
        logger.info(f"WalkForwardValidator initialized with in_sample_ratio={in_sample_ratio}, "
                   f"num_periods={num_periods}, min_period_length={min_period_length}")
    
    def validate_strategy(self,
                         strategy: Strategy,
                         data: pd.DataFrame,
                         metrics: List[str] = ['sharpe_ratio', 'max_drawdown', 'win_rate']) -> Dict:
        """
        Validate strategy using walk-forward optimization.
        
        Args:
            strategy: Strategy to validate
            data: Historical data for validation
            metrics: List of performance metrics to track
            
        Returns:
            Dict: Validation results including consistency metrics
        """
        if len(data) < self.min_period_length:
            raise ValueError(f"Data length {len(data)} is less than minimum period length {self.min_period_length}")
        
        # Split data into periods
        periods = self._split_data(data)
        
        # Track results for each period
        results = []
        for i, (in_sample, out_of_sample) in enumerate(periods):
            logger.info(f"Validating period {i+1}/{len(periods)}")
            
            # Optimize strategy on in-sample data
            optimized_strategy = self._optimize_strategy(strategy, in_sample)
            
            # Test strategy on out-of-sample data
            period_results = self._test_strategy(optimized_strategy, out_of_sample, metrics)
            results.append(period_results)
        
        # Calculate consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(results)
        
        # Aggregate results
        validation_results = {
            'period_results': results,
            'consistency_metrics': consistency_metrics,
            'overall_performance': self._aggregate_performance(results)
        }
        
        logger.info(f"Walk-forward validation completed with {len(results)} periods")
        return validation_results
    
    def _split_data(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split data into in-sample and out-of-sample periods"""
        total_length = len(data)
        period_length = total_length // self.num_periods
        
        if period_length < self.min_period_length:
            raise ValueError(f"Period length {period_length} is less than minimum {self.min_period_length}")
        
        periods = []
        for i in range(self.num_periods):
            start_idx = i * period_length
            end_idx = start_idx + period_length
            
            if i == self.num_periods - 1:
                end_idx = total_length
            
            period_data = data.iloc[start_idx:end_idx]
            in_sample_length = int(len(period_data) * self.in_sample_ratio)
            
            in_sample = period_data.iloc[:in_sample_length]
            out_of_sample = period_data.iloc[in_sample_length:]
            
            periods.append((in_sample, out_of_sample))
        
        return periods
    
    def _optimize_strategy(self, strategy: Strategy, data: pd.DataFrame) -> Strategy:
        """Optimize strategy parameters on in-sample data"""
        # Create a copy of the strategy
        optimized_strategy = strategy.copy()
        
        # Optimize parameters
        # Implementation depends on your optimization method
        # This could involve grid search, genetic algorithms, etc.
        
        return optimized_strategy
    
    def _test_strategy(self,
                      strategy: Strategy,
                      data: pd.DataFrame,
                      metrics: List[str]) -> Dict:
        """Test strategy on out-of-sample data"""
        # Run strategy on data
        results = strategy.run(data)
        
        # Calculate performance metrics
        performance = {}
        for metric in metrics:
            if metric == 'sharpe_ratio':
                performance[metric] = self._calculate_sharpe_ratio(results['returns'])
            elif metric == 'max_drawdown':
                performance[metric] = self._calculate_max_drawdown(results['equity'])
            elif metric == 'win_rate':
                performance[metric] = self._calculate_win_rate(results['trades'])
            # Add more metrics as needed
        
        return {
            'performance': performance,
            'trades': results['trades'],
            'equity_curve': results['equity']
        }
    
    def _calculate_consistency_metrics(self, results: List[Dict]) -> Dict:
        """Calculate consistency metrics across all periods"""
        # Extract performance metrics for each period
        metrics = {}
        for result in results:
            for metric, value in result['performance'].items():
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(value)
        
        # Calculate consistency metrics
        consistency = {}
        for metric, values in metrics.items():
            consistency[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'positive_periods': sum(1 for v in values if v > 0) / len(values)
            }
        
        return consistency
    
    def _aggregate_performance(self, results: List[Dict]) -> Dict:
        """Aggregate performance across all periods"""
        # Combine all trades
        all_trades = []
        for result in results:
            all_trades.extend(result['trades'])
        
        # Calculate overall performance metrics
        return {
            'total_trades': len(all_trades),
            'win_rate': self._calculate_win_rate(all_trades),
            'profit_factor': self._calculate_profit_factor(all_trades),
            'avg_trade': np.mean([t['profit'] for t in all_trades]) if all_trades else 0
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        return abs(drawdowns.min())
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate"""
        if not trades:
            return 0.0
        winning_trades = sum(1 for t in trades if t['profit'] > 0)
        return winning_trades / len(trades)
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0.0
        gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in trades if t['profit'] < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf') 