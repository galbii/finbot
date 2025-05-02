import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GrammarParser')

class GrammarParser:
    """
    Parser for trading strategy grammar.
    
    This class is responsible for:
    - Parsing grammar strings into executable components
    - Validating grammar syntax
    - Converting grammar rules to condition functions
    """
    
    def __init__(self, grammar_file=None):
        """
        Initialize the grammar parser.
        
        Args:
            grammar_file (str, optional): Path to a grammar definition file
        """
        self.grammar = None
        if grammar_file:
            self._load_grammar(grammar_file)
        logger.info("GrammarParser initialized")
    
    def _load_grammar(self, grammar_file):
        """
        Load grammar from a file.
        
        Args:
            grammar_file (str): Path to grammar file
        """
        try:
            with open(grammar_file, 'r') as f:
                # This is a simplification; actual implementation would parse the grammar file
                self.grammar = {}
            logger.info(f"Grammar loaded from {grammar_file}")
        except Exception as e:
            logger.error(f"Failed to load grammar: {e}")
    
    def parse(self, grammar_string):
        """
        Parse a grammar string into executable components.
        
        Args:
            grammar_string (str): Strategy string following the grammar
            
        Returns:
            dict: Parsed strategy components
        """
        logger.info(f"Parsing grammar string: {grammar_string[:50]}...")
        
        try:
            # Split into entry and exit rules (and optional position sizing)
            parts = grammar_string.split("SELL WHEN")
            
            if len(parts) != 2:
                raise ValueError("Grammar string must contain 'BUY WHEN' and 'SELL WHEN' sections")
            
            entry_part = parts[0].strip()
            exit_part = "SELL WHEN " + parts[1].strip()
            
            # Check for position sizing
            position_sizing = None
            if "SIZE" in exit_part:
                size_parts = exit_part.split("SIZE")
                exit_part = size_parts[0].strip()
                position_sizing = "SIZE" + size_parts[1].strip()
            
            # Parse entry rules
            if not entry_part.startswith("BUY WHEN"):
                raise ValueError("Entry rule must start with 'BUY WHEN'")
            
            entry_conditions = self._parse_conditions(entry_part[8:].strip())
            exit_conditions = self._parse_conditions(exit_part[9:].strip())
            
            parsed_position_sizing = self._parse_position_sizing(position_sizing) if position_sizing else None
            
            # Create and return the parsed strategy
            parsed_strategy = {
                'entry_conditions': entry_conditions,
                'exit_conditions': exit_conditions,
                'position_sizing': parsed_position_sizing
            }
            
            logger.info("Grammar string successfully parsed")
            return parsed_strategy
        
        except Exception as e:
            logger.error(f"Failed to parse grammar string: {e}")
            raise
    
    def _parse_conditions(self, conditions_str):
        """
        Parse condition string into a structured representation.
        
        Args:
            conditions_str (str): Condition string (e.g., "RSI(14) < 30")
            
        Returns:
            dict: Parsed conditions with their components
        """
        try:
            # Handle empty conditions
            if not conditions_str.strip():
                raise ValueError("Empty condition string")
            
            # Split conditions if there are AND/OR operators
            if " AND " in conditions_str:
                subconditions = conditions_str.split(" AND ")
                parsed_subconditions = [self._parse_single_condition(sub.strip()) for sub in subconditions]
                return {
                    'operator': 'AND',
                    'conditions': parsed_subconditions
                }
            elif " OR " in conditions_str:
                subconditions = conditions_str.split(" OR ")
                parsed_subconditions = [self._parse_single_condition(sub.strip()) for sub in subconditions]
                return {
                    'operator': 'OR',
                    'conditions': parsed_subconditions
                }
            else:
                # Single condition - wrap it in a compound condition structure
                parsed_condition = self._parse_single_condition(conditions_str.strip())
                return {
                    'operator': 'AND',  # Default to AND for single conditions
                    'conditions': [parsed_condition]
                }
            
        except Exception as e:
            logger.error(f"Error parsing conditions: {e}")
            raise
    
    def _parse_single_condition(self, condition_str):
        """
        Parse a single condition into its components.
        
        Args:
            condition_str (str): Single condition string (e.g., "RSI(14) < 30")
            
        Returns:
            dict: Parsed condition components
        """
        try:
            # Handle MACD crosses
            if "CROSSES" in condition_str:
                if "CROSSES ABOVE" in condition_str:
                    parts = condition_str.split("CROSSES ABOVE")
                    operator = "CROSSES ABOVE"
                else:
                    parts = condition_str.split("CROSSES BELOW")
                    operator = "CROSSES BELOW"
                indicator = parts[0].strip()
                reference = parts[1].strip()
            else:
                # Handle other conditions
                for op in [" < ", " > ", " <= ", " >= ", " == "]:
                    if op in condition_str:
                        parts = condition_str.split(op)
                        operator = op.strip()
                        indicator = parts[0].strip()
                        reference = parts[1].strip()
                        break
                else:
                    raise ValueError(f"No valid operator found in condition: {condition_str}")
            
            # Parse indicator parameters if present
            params = []
            if "(" in indicator:
                indicator_name = indicator[:indicator.find("(")]
                params_str = indicator[indicator.find("(")+1:indicator.find(")")]
                params = [p.strip() for p in params_str.split(",")]
            else:
                indicator_name = indicator
            
            # Handle special cases for Bollinger Bands and Price conditions
            if indicator_name == "PRICE":
                if "BB_" in reference:
                    indicator_name = "BOLLINGER_BANDS"
                    # The reference is already in the correct format (e.g., BB_upper_20_2.0)
                    # No need to modify it
                    params = reference.split("_")[2:]  # Extract window and std from the reference
            
            # Return a consistent structure for all conditions
            return {
                'type': 'CONDITION',
                'indicator': indicator_name,
                'parameters': params,
                'operator': operator,
                'reference': reference
            }
            
        except Exception as e:
            logger.error(f"Error parsing single condition: {e}")
            raise
    
    def _parse_position_sizing(self, position_sizing_str):
        """
        Parse position sizing rule.
        
        Args:
            position_sizing_str (str): Position sizing string
            
        Returns:
            dict: Parsed position sizing rule
        """
        if not position_sizing_str:
            return None
        
        # Parse percentage (check this first since it's more specific)
        percentage_match = re.match(r'SIZE\s*=\s*(\d+)%', position_sizing_str)
        if percentage_match:
            percentage = percentage_match.group(1)
            return {
                'type': 'PERCENTAGE',
                'percentage': float(percentage)
            }
        
        # Parse fixed amount
        fixed_match = re.match(r'SIZE\s*=\s*(\d+)', position_sizing_str)
        if fixed_match:
            amount = fixed_match.group(1)
            return {
                'type': 'FIXED',
                'amount': float(amount)
            }
        
        # Parse risk-based
        risk_match = re.match(r'SIZE\s*=\s*(\d+)%\s+OF PORTFOLIO', position_sizing_str)
        if risk_match:
            risk = risk_match.group(1)
            return {
                'type': 'RISK_BASED',
                'risk_percentage': float(risk)
            }
        
        # If no match, return default
        logger.warning(f"Could not parse position sizing: {position_sizing_str}. Using default.")
        return {
            'type': 'PERCENTAGE',
            'percentage': 100.0
        }
    
    def validate(self, grammar_string):
        """
        Validate a grammar string against the grammar rules.
        
        Args:
            grammar_string (str): Strategy string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            self.parse(grammar_string)
            return True
        except Exception as e:
            logger.warning(f"Grammar validation failed: {e}")
            return False
    
    def generate_condition_function(self, condition_dict):
        """
        Generate a function that evaluates a condition or compound condition.
        
        Args:
            condition_dict (dict): Parsed condition dictionary
            
        Returns:
            callable: Function that takes a date and returns a boolean
        """
        try:
            # Handle compound conditions
            if 'operator' in condition_dict and 'conditions' in condition_dict:
                subconditions = [self.generate_condition_function(cond) for cond in condition_dict['conditions']]
                operator = condition_dict['operator']
                
                def evaluate_compound(date):
                    try:
                        if operator == 'AND':
                            return all(cond(date) for cond in subconditions)
                        elif operator == 'OR':
                            return any(cond(date) for cond in subconditions)
                        else:
                            raise ValueError(f"Unknown operator: {operator}")
                    except Exception as e:
                        logger.error(f"Error evaluating compound condition: {e}")
                        return False
                
                return evaluate_compound
            
            # Handle single conditions
            if condition_dict.get('type') != 'CONDITION':
                raise ValueError(f"Invalid condition type: {condition_dict.get('type')}")
            
            indicator = condition_dict['indicator']
            operator = condition_dict['operator']
            params = condition_dict['parameters']
            reference = condition_dict['reference']
            
            if indicator == 'RSI':
                period = int(params[0]) if params else 14
                threshold = float(reference)
                
                def evaluate_rsi(date):
                    try:
                        rsi_value = self.indicator_calculator.calculate_rsi(date, period)
                        if operator == '>':
                            return rsi_value > threshold
                        elif operator == '<':
                            return rsi_value < threshold
                        elif operator == 'CROSSES ABOVE':
                            return self.indicator_calculator.crosses_above(
                                lambda d: self.indicator_calculator.calculate_rsi(d, period),
                                lambda d: threshold,
                                date
                            )
                        elif operator == 'CROSSES BELOW':
                            return self.indicator_calculator.crosses_below(
                                lambda d: self.indicator_calculator.calculate_rsi(d, period),
                                lambda d: threshold,
                                date
                            )
                        else:
                            raise ValueError(f"Unknown operator for RSI: {operator}")
                    except Exception as e:
                        logger.error(f"Error evaluating RSI condition: {e}")
                        return False
                
                return evaluate_rsi
                
            elif indicator == 'MACD':
                fast_period = int(params[0]) if len(params) > 0 else 12
                slow_period = int(params[1]) if len(params) > 1 else 26
                signal_period = int(params[2]) if len(params) > 2 else 9
                
                def evaluate_macd(date):
                    try:
                        macd_line, signal_line = self.indicator_calculator.calculate_macd(
                            date, fast_period, slow_period, signal_period
                        )
                        if operator == 'CROSSES ABOVE':
                            return self.indicator_calculator.crosses_above(
                                lambda d: self.indicator_calculator.calculate_macd(d, fast_period, slow_period, signal_period)[0],
                                lambda d: self.indicator_calculator.calculate_macd(d, fast_period, slow_period, signal_period)[1],
                                date
                            )
                        elif operator == 'CROSSES BELOW':
                            return self.indicator_calculator.crosses_below(
                                lambda d: self.indicator_calculator.calculate_macd(d, fast_period, slow_period, signal_period)[0],
                                lambda d: self.indicator_calculator.calculate_macd(d, fast_period, slow_period, signal_period)[1],
                                date
                            )
                        else:
                            raise ValueError(f"Unknown operator for MACD: {operator}")
                    except Exception as e:
                        logger.error(f"Error evaluating MACD condition: {e}")
                        return False
                
                return evaluate_macd
                
            elif indicator == 'BOLLINGER_BANDS':
                window = int(params[0]) if len(params) > 0 else 20
                std = float(params[1]) if len(params) > 1 else 2.0
                
                def evaluate_bb(date):
                    try:
                        upper_band, middle_band, lower_band = self.indicator_calculator.calculate_bollinger_bands(
                            date, window, std
                        )
                        price = self.indicator_calculator.get_price(date)
                        
                        if reference == 'UPPER':
                            band = upper_band
                        elif reference == 'LOWER':
                            band = lower_band
                        else:
                            raise ValueError(f"Invalid Bollinger Band reference: {reference}")
                            
                        if operator == 'CROSSES ABOVE':
                            return self.indicator_calculator.crosses_above(
                                lambda d: self.indicator_calculator.get_price(d),
                                lambda d: band,
                                date
                            )
                        elif operator == 'CROSSES BELOW':
                            return self.indicator_calculator.crosses_below(
                                lambda d: self.indicator_calculator.get_price(d),
                                lambda d: band,
                                date
                            )
                        elif operator == '>':
                            return price > band
                        elif operator == '<':
                            return price < band
                        else:
                            raise ValueError(f"Unknown operator for Bollinger Bands: {operator}")
                    except Exception as e:
                        logger.error(f"Error evaluating Bollinger Bands condition: {e}")
                        return False
                
                return evaluate_bb
            
            else:
                raise ValueError(f"Unknown indicator: {indicator}")
                
        except Exception as e:
            logger.error(f"Error generating condition function: {e}")
            raise 