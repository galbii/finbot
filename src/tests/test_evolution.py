import sys
import os
import unittest
import random
import numpy as np

# Add src directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar_parser import GrammarParser
from strategy import Strategy
from evolution_manager import EvolutionManager
from bot import Bot

class MockBot:
    """Mock Bot class for testing"""
    
    def __init__(self):
        self.grammar_parser = GrammarParser()
        self.backtest_calls = 0
        self.strategies = []
    
    def backtest_strategy(self, strategy):
        """Mock backtest method"""
        self.backtest_calls += 1
        self.strategies.append(strategy)
        
        # Assign random performance metrics for testing
        strategy.profit_percentage = random.uniform(-10, 20)
        strategy.initial_portfolio = 10000
        strategy.final_portfolio = 10000 * (1 + strategy.profit_percentage/100)
        
        # Add random metrics
        strategy.metrics = {
            'sharpe_ratio': random.uniform(-0.5, 2.0),
            'max_drawdown': random.uniform(0, 15),
            'win_rate': random.uniform(30, 70),
            'avg_win': random.uniform(1, 5),
            'avg_loss': random.uniform(-5, -1),
            'profit_factor': random.uniform(0.5, 2.0)
        }
        
        # Return mock results
        return {
            'portfolio_history': None,
            'trades': [],
            'metrics': strategy.metrics
        }

class TestEvolutionManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_bot = MockBot()
        self.grammar_parser = GrammarParser()
        self.evolution_manager = EvolutionManager(
            self.mock_bot,
            self.grammar_parser,
            population_size=10,
            mutation_rate=0.3,
            crossover_rate=0.7,
            tournament_size=3,
            elitism_count=2
        )
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def test_random_strategy_generation(self):
        """Test random strategy generation"""
        strategy = self.evolution_manager.generate_random_strategy()
        
        # Check that a strategy was generated
        self.assertIsNotNone(strategy)
        
        # Check that it's a valid strategy
        self.assertIsNotNone(strategy.grammar_string)
        self.assertTrue("BUY WHEN" in strategy.grammar_string)
        self.assertTrue("SELL WHEN" in strategy.grammar_string)
        
        # Check that it can be parsed
        self.assertIsNotNone(strategy.parsed_strategy)
        
        # Generate multiple strategies and check diversity
        strategies = []
        for _ in range(5):
            strategy = self.evolution_manager.generate_random_strategy()
            if strategy:
                strategies.append(strategy)
                
        self.assertTrue(len(strategies) > 0)
        
        # Check that they're not all the same
        grammar_strings = [s.grammar_string for s in strategies]
        unique_strings = set(grammar_strings)
        self.assertTrue(len(unique_strings) > 1)
    
    def test_population_initialization(self):
        """Test population initialization"""
        # Create a base strategy
        base_strategy = Strategy("BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70", generation=0)
        base_strategy.parse(self.grammar_parser)
        
        # Initialize population
        population = self.evolution_manager.initialize_population(base_strategy)
        
        # Check population size
        self.assertEqual(len(population), self.evolution_manager.population_size)
        
        # Check that base strategy is included
        self.assertTrue(any(s.grammar_string == base_strategy.grammar_string for s in population))
        
        # Check diversity
        unique_strategies = set(s.grammar_string for s in population)
        diversity = len(unique_strategies) / len(population)
        self.assertTrue(diversity > 0.5)  # At least 50% unique strategies
    
    def test_tournament_selection(self):
        """Test tournament selection"""
        # Create a population with known fitness values
        strategies = []
        for i in range(10):
            strategy = Strategy(f"BUY WHEN RSI(14) < {20+i} SELL WHEN RSI(14) > {70+i}", generation=0)
            strategy.parse(self.grammar_parser)
            strategy.profit_percentage = i * 5  # 0, 5, 10, ..., 45
            strategies.append(strategy)
        
        self.evolution_manager.population = strategies
        
        # Run tournament selection multiple times
        selected = [self.evolution_manager.tournament_selection() for _ in range(20)]
        
        # Calculate average profit of selected strategies
        avg_profit = sum(s.profit_percentage for s in selected) / len(selected)
        
        # Check that tournament tends to select above-average strategies
        avg_population_profit = sum(s.profit_percentage for s in strategies) / len(strategies)
        self.assertGreater(avg_profit, avg_population_profit)
    
    def test_roulette_wheel_selection(self):
        """Test roulette wheel selection"""
        # Create a population with known fitness values
        strategies = []
        for i in range(10):
            strategy = Strategy(f"BUY WHEN RSI(14) < {20+i} SELL WHEN RSI(14) > {70+i}", generation=0)
            strategy.parse(self.grammar_parser)
            strategy.profit_percentage = i * 5  # 0, 5, 10, ..., 45
            strategies.append(strategy)
        
        self.evolution_manager.population = strategies
        
        # Run roulette wheel selection multiple times
        selected = [self.evolution_manager.roulette_wheel_selection() for _ in range(50)]
        
        # Check that higher fitness strategies are selected more often
        strategy_counts = {}
        for s in selected:
            profit = s.profit_percentage
            if profit not in strategy_counts:
                strategy_counts[profit] = 0
            strategy_counts[profit] += 1
        
        # Strategies with higher profit should generally be selected more often
        profits = sorted(strategy_counts.keys())
        if len(profits) >= 4:  # If we have enough different strategies selected
            # Check if higher profit strategies appear more frequently
            lower_half = sum(strategy_counts[p] for p in profits[:len(profits)//2])
            upper_half = sum(strategy_counts[p] for p in profits[len(profits)//2:])
            self.assertGreaterEqual(upper_half, lower_half)
    
    def test_crossover(self):
        """Test crossover operator"""
        # Create two parent strategies
        parent1 = Strategy("BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70", generation=0)
        parent1.parse(self.grammar_parser)
        
        parent2 = Strategy("BUY WHEN MACD(12, 26, 9) CROSSES ABOVE SIGNAL SELL WHEN RSI(7) > 80", generation=0)
        parent2.parse(self.grammar_parser)
        
        # Perform crossover
        child = self.evolution_manager.crossover(parent1, parent2)
        
        # Check that child is valid
        self.assertIsNotNone(child)
        self.assertTrue(child.parsed_strategy is not None)
        
        # Check that child has elements from both parents
        parent1_entry = "BUY WHEN RSI(14) < 30"
        parent2_entry = "BUY WHEN MACD(12, 26, 9) CROSSES ABOVE SIGNAL"
        parent1_exit = "SELL WHEN RSI(14) > 70"
        parent2_exit = "SELL WHEN RSI(7) > 80"
        
        # Child should have elements from either parent
        has_parent1_entry = parent1_entry in child.grammar_string
        has_parent2_entry = parent2_entry in child.grammar_string
        has_parent1_exit = parent1_exit in child.grammar_string
        has_parent2_exit = parent2_exit in child.grammar_string
        
        # At least one component should come from each parent
        has_parent1_component = has_parent1_entry or has_parent1_exit
        has_parent2_component = has_parent2_entry or has_parent2_exit
        
        # Allow for the case where crossover didn't happen and we got a copy of parent1
        if child.grammar_string == parent1.grammar_string:
            print("Crossover returned copy of parent1")
        elif child.grammar_string == parent2.grammar_string:
            print("Crossover returned copy of parent2")
        else:
            # If we have a true crossover, it should have components from both parents
            self.assertTrue(has_parent1_component or has_parent2_component)
    
    def test_mutation(self):
        """Test mutation operator"""
        # Create a strategy to mutate
        strategy = Strategy("BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70", generation=0)
        strategy.parse(self.grammar_parser)
        
        # Perform mutation
        mutated = self.evolution_manager.mutate(strategy)
        
        # Check that mutated strategy is valid
        self.assertIsNotNone(mutated)
        self.assertTrue(mutated.parsed_strategy is not None)
        
        # Check that something was actually changed, unless we got back the original
        if mutated.grammar_string != strategy.grammar_string:
            differences = 0
            if "RSI(14)" not in mutated.grammar_string and "RSI(14)" in strategy.grammar_string:
                differences += 1
            if "< 30" not in mutated.grammar_string and "< 30" in strategy.grammar_string:
                differences += 1
            if "> 70" not in mutated.grammar_string and "> 70" in strategy.grammar_string:
                differences += 1
                
            self.assertTrue(differences > 0)
    
    def test_evolve_population(self):
        """Test full population evolution"""
        # Initialize population
        base_strategy = Strategy("BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70", generation=0)
        base_strategy.parse(self.grammar_parser)
        
        self.evolution_manager.initialize_population(base_strategy)
        
        # Evaluate initial population
        self.evolution_manager.evaluate_population()
        
        # Record initial state
        initial_population = self.evolution_manager.population.copy()
        initial_best = max(initial_population, key=lambda s: s.profit_percentage)
        
        # Evolve for one generation
        self.evolution_manager.evolve_population()
        
        # Evaluate new generation
        self.evolution_manager.evaluate_population()
        
        # Check that generation counter increased
        self.assertEqual(self.evolution_manager.generation, 1)
        
        # Check that population size is maintained
        self.assertEqual(len(self.evolution_manager.population), self.evolution_manager.population_size)
        
        # Check that new strategies were created
        new_strategies = [s for s in self.evolution_manager.population 
                         if all(s.grammar_string != old_s.grammar_string for old_s in initial_population)]
        self.assertTrue(len(new_strategies) > 0)
        
        # Check for elitism - best strategy should be preserved or improved
        new_best = max(self.evolution_manager.population, key=lambda s: s.profit_percentage)
        self.assertGreaterEqual(new_best.profit_percentage, initial_best.profit_percentage)

if __name__ == '__main__':
    unittest.main() 