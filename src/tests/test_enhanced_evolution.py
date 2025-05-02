import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enhanced_evolution_manager import EnhancedEvolutionManager
from strategy import Strategy
from market_regime_detector import MarketRegime, MarketRegimeDetector
from walk_forward_validator import WalkForwardValidator
from multi_objective_fitness import MultiObjectiveFitness
from strategy_archive import StrategyArchive

class TestEnhancedEvolutionManager(unittest.TestCase):
    """Test cases for EnhancedEvolutionManager"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate price data with trend and noise
        trend = np.linspace(100, 200, len(dates))
        noise = np.random.normal(0, 2, len(dates))
        prices = trend + noise
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 2, len(dates)),
            'low': prices - np.random.uniform(0, 2, len(dates)),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Create grammar
        self.grammar = {
            'start': 'strategy',
            'strategy': [
                'if {condition} then {action}',
                'if {condition} then {action} else {action}'
            ],
            'condition': [
                '{indicator} {operator} {value}',
                '{condition} and {condition}',
                '{condition} or {condition}'
            ],
            'action': [
                'buy',
                'sell',
                'hold'
            ],
            'indicator': [
                'sma_20',
                'sma_50',
                'rsi',
                'macd',
                'bollinger_bands',
                'atr'
            ],
            'operator': [
                '>',
                '<',
                '>=',
                '<=',
                '=='
            ],
            'value': [
                '0.0',
                '0.1',
                '0.2',
                '0.3',
                '0.4',
                '0.5',
                '0.6',
                '0.7',
                '0.8',
                '0.9',
                '1.0'
            ]
        }
        
        # Create config
        self.config = {
            'population_size': 10,
            'generations': 5,
            'tournament_size': 3,
            'objectives': ['sharpe_ratio', 'max_drawdown', 'win_rate'],
            'weights': {
                'sharpe_ratio': 0.4,
                'max_drawdown': 0.3,
                'win_rate': 0.3
            },
            'min_trades': 5,
            'regime_lookback': 20,
            'volatility_threshold': 0.02,
            'trend_threshold': 0.1,
            'in_sample_ratio': 0.7,
            'num_periods': 3,
            'min_period_length': 50,
            'archive_size': 20,
            'similarity_threshold': 0.2,
            'archive_threshold': 0.7
        }
    
    def test_initialization(self):
        """Test initialization of EnhancedEvolutionManager"""
        manager = EnhancedEvolutionManager(
            config=self.config,
            data=self.data,
            grammar=self.grammar
        )
        
        self.assertIsInstance(manager.fitness_evaluator, MultiObjectiveFitness)
        self.assertIsInstance(manager.regime_detector, MarketRegimeDetector)
        self.assertIsInstance(manager.validator, WalkForwardValidator)
        self.assertIsInstance(manager.strategy_archive, StrategyArchive)
        self.assertEqual(len(manager.population), 0)
        self.assertEqual(manager.generation, 0)
    
    def test_initialize_population(self):
        """Test population initialization"""
        manager = EnhancedEvolutionManager(
            config=self.config,
            data=self.data,
            grammar=self.grammar
        )
        
        manager.initialize_population(self.config['population_size'])
        
        self.assertEqual(len(manager.population), self.config['population_size'])
        for strategy in manager.population:
            self.assertIsInstance(strategy, Strategy)
            self.assertIsNotNone(strategy.grammar_string)
    
    def test_evaluate_population(self):
        """Test population evaluation"""
        manager = EnhancedEvolutionManager(
            config=self.config,
            data=self.data,
            grammar=self.grammar
        )
        
        manager.initialize_population(self.config['population_size'])
        manager.evaluate_population()
        
        for strategy in manager.population:
            self.assertIsNotNone(strategy.fitness)
            self.assertIn('composite', strategy.fitness)
            self.assertGreaterEqual(strategy.fitness['composite'], 0)
            self.assertLessEqual(strategy.fitness['composite'], 1)
    
    def test_evolve_population(self):
        """Test population evolution"""
        manager = EnhancedEvolutionManager(
            config=self.config,
            data=self.data,
            grammar=self.grammar
        )
        
        manager.initialize_population(self.config['population_size'])
        manager.evaluate_population()
        
        initial_fitness = [s.fitness['composite'] for s in manager.population]
        
        manager.evolve_population()
        
        self.assertEqual(len(manager.population), self.config['population_size'])
        self.assertEqual(manager.generation, 1)
        
        final_fitness = [s.fitness['composite'] for s in manager.population]
        
        # Check that at least some strategies improved
        self.assertTrue(any(f > i for f, i in zip(final_fitness, initial_fitness)))
    
    def test_run_evolution(self):
        """Test full evolution process"""
        manager = EnhancedEvolutionManager(
            config=self.config,
            data=self.data,
            grammar=self.grammar
        )
        
        best_strategy = manager.run_evolution(
            num_generations=self.config['generations'],
            population_size=self.config['population_size']
        )
        
        self.assertIsInstance(best_strategy, Strategy)
        self.assertIsNotNone(best_strategy.fitness)
        self.assertIn('composite', best_strategy.fitness)
        self.assertGreaterEqual(best_strategy.fitness['composite'], 0)
        self.assertLessEqual(best_strategy.fitness['composite'], 1)
    
    def test_market_regime_detection(self):
        """Test market regime detection"""
        manager = EnhancedEvolutionManager(
            config=self.config,
            data=self.data,
            grammar=self.grammar
        )
        
        regime, scores = manager.regime_detector.detect_regime(
            self.data['close'],
            self.data['volume']
        )
        
        self.assertIsInstance(regime, MarketRegime)
        self.assertIsInstance(scores, dict)
        self.assertTrue(all(isinstance(k, MarketRegime) for k in scores.keys()))
        self.assertTrue(all(isinstance(v, float) for v in scores.values()))
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)
    
    def test_walk_forward_validation(self):
        """Test walk-forward validation"""
        manager = EnhancedEvolutionManager(
            config=self.config,
            data=self.data,
            grammar=self.grammar
        )
        
        # Create a test strategy
        strategy = Strategy("if rsi < 30 then buy else if rsi > 70 then sell")
        
        # Run validation
        results = manager.validator.validate_strategy(strategy, self.data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('period_results', results)
        self.assertIn('consistency_metrics', results)
        self.assertIn('overall_performance', results)
        
        # Check period results
        self.assertEqual(len(results['period_results']), self.config['num_periods'])
        for period_result in results['period_results']:
            self.assertIn('performance', period_result)
            self.assertIn('trades', period_result)
            self.assertIn('equity_curve', period_result)
    
    def test_strategy_archive(self):
        """Test strategy archiving"""
        manager = EnhancedEvolutionManager(
            config=self.config,
            data=self.data,
            grammar=self.grammar
        )
        
        # Create some test strategies
        strategies = [
            Strategy("if rsi < 30 then buy else if rsi > 70 then sell"),
            Strategy("if sma_20 > sma_50 then buy else sell"),
            Strategy("if macd > 0 then buy else sell")
        ]
        
        # Add strategies to archive
        for strategy in strategies:
            fitness = {'composite': 0.8}  # Good enough to be archived
            self.assertTrue(manager.strategy_archive.add_strategy(strategy, fitness))
        
        # Get diverse seeds
        seeds = manager.strategy_archive.get_diverse_seeds(n=2)
        
        self.assertEqual(len(seeds), 2)
        self.assertTrue(all(isinstance(s, Strategy) for s in seeds))
        
        # Check archive summary
        summary = manager.strategy_archive.get_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('size', summary)
        self.assertIn('avg_fitness', summary)
        self.assertIn('diversity', summary)

if __name__ == '__main__':
    unittest.main() 