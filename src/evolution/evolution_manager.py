import random
import logging
import numpy as np
from src.core.strategy import Strategy
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import os
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EvolutionManager')

class EvolutionManager:
    """
    Manages the evolutionary process for trading strategies.
    
    Responsibilities:
    - Generating random valid strategies
    - Evaluating fitness of strategies
    - Managing population across generations
    - Implementing genetic operators (selection, crossover, mutation)
    - Tracking diversity and improvement
    """
    
    def __init__(self, bot, grammar_parser, population_size=50, mutation_rate=0.3, 
                 crossover_rate=0.7, tournament_size=3, elitism_count=2, 
                 num_workers=None, batch_size=10, early_termination=True):
        """
        Initialize the evolution manager.
        
        Args:
            bot: Bot instance for strategy evaluation
            grammar_parser: GrammarParser for strategy validation
            population_size (int): Size of the population
            mutation_rate (float): Probability of mutation (0-1)
            crossover_rate (float): Probability of crossover (0-1)
            tournament_size (int): Number of strategies in tournament selection
            elitism_count (int): Number of top strategies to preserve
            num_workers (int, optional): Number of worker processes for parallel evaluation
                                        If None, uses max(1, CPU count - 1)
            batch_size (int): Number of strategies to evaluate in a batch
            early_termination (bool): Whether to enable early termination for poor strategies
        """
        self.bot = bot
        self.grammar_parser = grammar_parser
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        # Set number of workers for parallel processing
        if num_workers is None:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_workers = num_workers
            
        self.batch_size = batch_size
        self.early_termination = early_termination
        
        # Initial population will be stored here
        self.population = []
        self.generation = 0
        self.best_strategy = None
        
        # Adaptive mutation parameters
        self.adaptive_mutation = False
        self.min_mutation_rate = 0.1
        self.max_mutation_rate = 0.5
        self.diversity_threshold = 0.2  # Below this diversity, increase mutation rate
        
        # Complexity control parameters
        self.complexity_control = False
        self.max_conditions = 3
        self.complexity_penalty = 0.05  # Penalty per condition above threshold
        
        # Specialized mutation types
        self.specialized_mutations = True
        self.mutation_types = {
            'parameter': 0.5,    # Change indicator parameters
            'threshold': 0.3,    # Change threshold values
            'operator': 0.1,     # Change operators
            'structure': 0.1     # Add/remove conditions
        }
        
        # Indicator parameters for mutation
        self.indicator_params = {
            'RSI': {
                'period': [7, 9, 14, 21, 30]
            },
            'MACD': {
                'fast_period': [8, 12, 16, 20],
                'slow_period': [21, 26, 30, 35],
                'signal_period': [5, 7, 9, 12]
            },
            'BOLLINGER_BANDS': {
                'window': [10, 15, 20, 30],
                'std': [1.5, 2.0, 2.5, 3.0]
            }
        }
        
        # Threshold values for mutation
        self.thresholds = {
            'RSI': [20, 25, 30, 35, 40, 60, 65, 70, 75, 80],
            'BB': [0.5, 0.75, 1.0, 1.25, 1.5]
        }
        
        # Position sizing options
        self.position_sizing = [
            "SIZE = 25%", 
            "SIZE = 50%", 
            "SIZE = 75%", 
            "SIZE = 100%",
            "SIZE = 25% OF PORTFOLIO",
            "SIZE = 10% OF PORTFOLIO"
        ]
        
        # Logical operators
        self.logical_operators = ["AND", "OR"]
        
        # Comparison operators
        self.comparison_operators = ["<", ">", "CROSSES ABOVE", "CROSSES BELOW"]
        
        logger.info("EvolutionManager initialized with population size "
                   f"{population_size}, mutation rate {mutation_rate}, "
                   f"crossover rate {crossover_rate}, {self.num_workers} workers")
    
    def initialize_population(self, base_strategy=None):
        """
        Initialize the population with random strategies.
        
        Args:
            base_strategy (Strategy, optional): Base strategy to include
            
        Returns:
            list: Initial population of strategies
        """
        logger.info(f"Initializing population of {self.population_size} strategies")
        
        self.population = []
        
        # Add base strategy if provided
        if base_strategy:
            self.population.append(base_strategy)
            logger.info(f"Added base strategy: {base_strategy.grammar_string}")
        
        # Add random strategies
        while len(self.population) < self.population_size:
            strategy = self.generate_random_strategy()
            if strategy:
                self.population.append(strategy)
        
        logger.info(f"Population initialized with {len(self.population)} strategies")
        return self.population
    
    def generate_random_strategy(self):
        """
        Generate a random valid strategy.
        
        Returns:
            Strategy: A randomly generated valid strategy
        """
        # Choose indicator types
        entry_indicator = random.choice(list(self.indicator_params.keys()))
        exit_indicator = random.choice(list(self.indicator_params.keys()))
        
        # Generate strategy string
        try:
            if entry_indicator == "RSI":
                period = random.choice(self.indicator_params['RSI']['period'])
                operator = random.choice(self.comparison_operators[:2])  # Only < and > for RSI
                threshold = random.choice(self.thresholds['RSI'])
                entry_condition = f"RSI({period}) {operator} {threshold}"
            elif entry_indicator == "MACD":
                fast = random.choice(self.indicator_params['MACD']['fast_period'])
                slow = random.choice(self.indicator_params['MACD']['slow_period'])
                signal = random.choice(self.indicator_params['MACD']['signal_period'])
                operator = random.choice(self.comparison_operators[2:])  # Only CROSSES for MACD
                reference = "SIGNAL"
                entry_condition = f"MACD({fast}, {slow}, {signal}) {operator} {reference}"
            elif entry_indicator == "BOLLINGER_BANDS":
                window = random.choice(self.indicator_params['BOLLINGER_BANDS']['window'])
                std = random.choice(self.indicator_params['BOLLINGER_BANDS']['std'])
                operator = random.choice(self.comparison_operators)
                band = random.choice(["UPPER", "LOWER"])
                entry_condition = f"PRICE {operator} BB_{band.lower()}_{window}_{std}"
            
            # Similar logic for exit condition
            if exit_indicator == "RSI":
                period = random.choice(self.indicator_params['RSI']['period'])
                operator = random.choice(self.comparison_operators[:2])
                threshold = random.choice(self.thresholds['RSI'])
                exit_condition = f"RSI({period}) {operator} {threshold}"
            elif exit_indicator == "MACD":
                fast = random.choice(self.indicator_params['MACD']['fast_period'])
                slow = random.choice(self.indicator_params['MACD']['slow_period'])
                signal = random.choice(self.indicator_params['MACD']['signal_period'])
                operator = random.choice(self.comparison_operators[2:])
                reference = "SIGNAL"
                exit_condition = f"MACD({fast}, {slow}, {signal}) {operator} {reference}"
            elif exit_indicator == "BOLLINGER_BANDS":
                window = random.choice(self.indicator_params['BOLLINGER_BANDS']['window'])
                std = random.choice(self.indicator_params['BOLLINGER_BANDS']['std'])
                operator = random.choice(self.comparison_operators)
                band = random.choice(["UPPER", "LOWER"])
                exit_condition = f"PRICE {operator} BB_{band.lower()}_{window}_{std}"
            
            # Add position sizing
            position_size = random.choice(self.position_sizing)
            
            # Format the complete strategy string with required prefixes
            strategy_string = f"BUY WHEN {entry_condition}\nSELL WHEN {exit_condition}\n{position_size}"
            
            # Create and return the strategy
            strategy = Strategy(strategy_string)
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating random strategy: {e}")
            return None
    
    def evaluate_population(self):
        """
        Evaluate all strategies in the population using parallel processing.
        
        Returns:
            list: Sorted population (best first)
        """
        logger.info(f"Evaluating population of {len(self.population)} strategies with {self.num_workers} workers")
        
        # Filter out already evaluated strategies
        to_evaluate = [s for s in self.population if s.final_portfolio <= 0]
        already_evaluated = [s for s in self.population if s.final_portfolio > 0]
        
        if not to_evaluate:
            logger.info("All strategies already evaluated")
            # Sort by profit percentage (descending)
            self.population.sort(key=lambda s: s.profit_percentage, reverse=True)
            return self.population
            
        logger.info(f"Evaluating {len(to_evaluate)} new strategies")
        
        # Set processed flag for child processes to access bot's data
        if not hasattr(self.bot, '_data_processed_for_multiprocessing'):
            # Prepare any data needed for multiprocessing
            # This is a hook for the bot to prepare its data for sharing across processes
            if hasattr(self.bot, 'prepare_for_multiprocessing'):
                self.bot.prepare_for_multiprocessing()
            self.bot._data_processed_for_multiprocessing = True
        
        # Process strategies in batches to avoid memory issues
        batches = [to_evaluate[i:i + self.batch_size] for i in range(0, len(to_evaluate), self.batch_size)]
        
        all_evaluated = already_evaluated.copy()
        
        start_time = time.time()
        
        for batch_num, batch in enumerate(batches):
            batch_start_time = time.time()
            logger.info(f"Processing batch {batch_num+1}/{len(batches)} with {len(batch)} strategies")
            
            if self.num_workers > 1:
                # Use ProcessPoolExecutor for parallelism
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    # Submit all strategies for evaluation
                    futures = []
                    for strategy in batch:
                        future = executor.submit(self._evaluate_strategy_wrapper, 
                                               strategy, 
                                               self.early_termination)
                        futures.append(future)
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            evaluated_strategy = future.result()
                            if evaluated_strategy:
                                all_evaluated.append(evaluated_strategy)
                                # Log progress
                                logger.debug(f"Strategy evaluated: {evaluated_strategy.profit_percentage:.2f}% profit")
                        except Exception as e:
                            logger.error(f"Error evaluating strategy: {e}")
            else:
                # Single-process evaluation for debugging or if parallel is disabled
                for strategy in batch:
                    try:
                        evaluated_strategy = self._evaluate_strategy_wrapper(
                            strategy, 
                            self.early_termination
                        )
                        if evaluated_strategy:
                            all_evaluated.append(evaluated_strategy)
                            logger.debug(f"Strategy evaluated: {evaluated_strategy.profit_percentage:.2f}% profit")
                    except Exception as e:
                        logger.error(f"Error evaluating strategy: {e}")
            
            batch_duration = time.time() - batch_start_time
            logger.info(f"Batch {batch_num+1} processed in {batch_duration:.2f} seconds")
        
        total_duration = time.time() - start_time
        logger.info(f"All strategies evaluated in {total_duration:.2f} seconds")
        
        # Update the population with all evaluated strategies
        self.population = all_evaluated
        
        # Sort by profit percentage (descending)
        self.population.sort(key=lambda s: s.profit_percentage, reverse=True)
        
        # Update best strategy
        if not self.best_strategy or self.population[0].profit_percentage > self.best_strategy.profit_percentage:
            self.best_strategy = copy.deepcopy(self.population[0])
            logger.info(f"New best strategy found: {self.best_strategy.profit_percentage:.2f}% profit")
        
        return self.population
    
    def _evaluate_strategy_wrapper(self, strategy, early_termination=True):
        """
        Wrapper for strategy evaluation to be used with multiprocessing.
        
        Args:
            strategy: Strategy to evaluate
            early_termination: Whether to enable early termination for poor strategies
            
        Returns:
            Strategy: Evaluated strategy
        """
        try:
            # Run backtest using the bot
            results = self.bot.backtest_strategy(strategy, early_termination=early_termination)
            
            if not results:
                # Invalid strategy, assign zero fitness
                strategy.profit_percentage = -100
                
            return strategy
        except Exception as e:
            logger.error(f"Error in strategy evaluation: {e}")
            # Return the strategy with a poor fitness score
            strategy.profit_percentage = -100
            return strategy
    
    def evolve_population(self, population, fitness_scores):
        """Evolve the population using genetic operators"""
        try:
            # Sort population by fitness
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
            
            # Apply elitism - keep top 2 strategies
            new_population = sorted_population[:2]
            
            # Fill rest of population with offspring
            while len(new_population) < len(population):
                # Select parents using tournament selection
                parent1 = self.tournament_selection(sorted_population)
                parent2 = self.tournament_selection(sorted_population)
                
                # Apply crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Apply mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add children to new population
                new_population.extend([child1, child2])
            
            # Trim to original population size
            new_population = new_population[:len(population)]
            
            return new_population
            
        except Exception as e:
            logger.error(f"Error in population evolution: {str(e)}")
            return population  # Return original population if evolution fails
            
    def crossover(self, parent1, parent2):
        """Perform crossover between two strategies"""
        try:
            # Split strategies into entry and exit conditions
            parent1_entry, parent1_exit = parent1.split()
            parent2_entry, parent2_exit = parent2.split()
            
            if parent1_entry is None or parent2_entry is None:
                return parent1, parent2
                
            # Create children by swapping conditions
            child1 = Strategy(f"{parent1_entry} SELL WHEN {parent2_exit}", parent1.generation + 1)
            child2 = Strategy(f"{parent2_entry} SELL WHEN {parent1_exit}", parent2.generation + 1)
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"Error in crossover: {str(e)}")
            return parent1, parent2  # Return parents if crossover fails
            
    def mutate(self, strategy):
        """Apply mutation to a strategy"""
        try:
            # Split strategy into entry and exit conditions
            entry_condition, exit_condition = strategy.split()
            if entry_condition is None:
                return strategy
                
            # Randomly choose which part to mutate
            if random.random() < 0.5:
                # Mutate entry condition
                entry_condition = self.mutate_condition(entry_condition)
            else:
                # Mutate exit condition
                exit_condition = self.mutate_condition(exit_condition)
                
            # Create new strategy with mutated condition
            new_strategy = Strategy(f"{entry_condition} SELL WHEN {exit_condition}", strategy.generation + 1)
            return new_strategy
            
        except Exception as e:
            logger.error(f"Error in mutation: {str(e)}")
            return strategy  # Return original strategy if mutation fails
            
    def mutate_condition(self, condition):
        """Mutate a single condition"""
        try:
            # Parse condition into components
            match = re.match(r"(.*?)\s*([<>]=?|=)\s*(\d+(?:\.\d+)?)", condition)
            if not match:
                return condition
                
            indicator, operator, value = match.groups()
            
            # Randomly choose mutation type
            mutation_type = random.choice(['indicator', 'operator', 'value'])
            
            if mutation_type == 'indicator':
                # Change indicator parameters
                if 'RSI' in indicator:
                    period = random.choice([7, 9, 14, 21, 30])
                    indicator = f"RSI({period})"
                elif 'BB' in indicator:
                    window = random.choice([10, 20, 30])
                    std = random.choice([1.5, 2.0, 2.5])
                    indicator = f"BB_{window}_{std}"
                elif 'MACD' in indicator:
                    fast = random.choice([8, 12, 16, 20])
                    slow = random.choice([21, 26, 30, 35])
                    signal = random.choice([5, 7, 9, 12])
                    indicator = f"MACD({fast},{slow},{signal})"
                    
            elif mutation_type == 'operator':
                # Change comparison operator
                operator = random.choice(['<', '<=', '>', '>=', '='])
                
            elif mutation_type == 'value':
                # Change threshold value
                if 'RSI' in indicator:
                    value = random.randint(20, 80)
                elif 'BB' in indicator:
                    value = random.uniform(0.5, 3.0)
                elif 'MACD' in indicator:
                    value = random.uniform(-2.0, 2.0)
                    
            return f"{indicator} {operator} {value}"
            
        except Exception as e:
            logger.error(f"Error in condition mutation: {str(e)}")
            return condition  # Return original condition if mutation fails
    
    def tournament_selection(self, population):
        """
        Select a strategy using tournament selection.
        
        Returns:
            Strategy: Selected strategy
        """
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda s: s.profit_percentage)
    
    def roulette_wheel_selection(self):
        """
        Select a strategy using roulette wheel selection.
        
        Returns:
            Strategy: Selected strategy
        """
        # Calculate fitness sum
        fitness_sum = sum(max(0.1, s.profit_percentage + 100) for s in self.population)
        
        # If fitness sum is zero or negative, use uniform selection
        if fitness_sum <= 0:
            return random.choice(self.population)
        
        # Generate random value
        value = random.uniform(0, fitness_sum)
        
        # Find the strategy that corresponds to this value
        current = 0
        for strategy in self.population:
            current += max(0.1, strategy.profit_percentage + 100)
            if current >= value:
                return strategy
        
        # Fallback
        return self.population[0]
    
    def calculate_diversity(self):
        """
        Calculate diversity of the current population.
        
        Returns:
            float: Diversity score (0-1)
        """
        if len(self.population) <= 1:
            return 0.0
            
        # Count unique strategies
        unique_strategies = set()
        for strategy in self.population:
            unique_strategies.add(strategy.grammar_string)
        
        # Diversity is the ratio of unique strategies to population size
        diversity = len(unique_strategies) / len(self.population)
        return diversity
    
    def get_summary(self):
        """
        Get summary of the current evolutionary state.
        
        Returns:
            dict: Evolution summary
        """
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_strategy': self.best_strategy.grammar_string if self.best_strategy else None,
            'best_profit': self.best_strategy.profit_percentage if self.best_strategy else 0,
            'diversity': self.calculate_diversity(),
            'avg_profit': np.mean([s.profit_percentage for s in self.population]) if self.population else 0
        } 