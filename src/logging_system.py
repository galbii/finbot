import logging
import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import csv

class LoggingSystem:
    """
    Comprehensive logging system for the trading strategy system.
    
    Handles:
    - Logging to file and console
    - Strategy evolution history
    - Performance metrics
    - System operations and errors
    - Export to various formats
    """
    
    def __init__(self, output_dir='logs', log_level=logging.INFO):
        """
        Initialize the logging system.
        
        Args:
            output_dir (str): Directory for log files
            log_level (int): Logging level (e.g., logging.INFO)
        """
        self.output_dir = output_dir
        self.log_level = log_level
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Setup file paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.system_log_path = os.path.join(output_dir, f'system_{timestamp}.log')
        self.evolution_log_path = os.path.join(output_dir, f'evolution_{timestamp}.csv')
        self.performance_log_path = os.path.join(output_dir, f'performance_{timestamp}.json')
        
        # Configure system logger
        self.logger = logging.getLogger('TradingSystem')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers if any
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.system_log_path)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize evolution log with header
        with open(self.evolution_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Generation', 'Best Profit', 'Avg Profit', 
                'Diversity', 'Mutation Rate', 'Duration', 'Best Strategy'
            ])
        
        self.logger.info(f"Logging system initialized. Logs will be saved to {output_dir}")
    
    def log_evolution_step(self, generation, best_strategy, avg_profit, diversity, 
                          mutation_rate, duration):
        """
        Log a single evolution step.
        
        Args:
            generation (int): Current generation number
            best_strategy (Strategy): Best strategy in the generation
            avg_profit (float): Average profit of the population
            diversity (float): Population diversity measure
            mutation_rate (float): Current mutation rate
            duration (float): Duration of the generation in seconds
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Log to evolution CSV
        with open(self.evolution_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, 
                generation, 
                best_strategy.profit_percentage, 
                avg_profit, 
                diversity, 
                mutation_rate, 
                duration,
                best_strategy.grammar_string
            ])
        
        # Log to system log
        self.logger.info(
            f"Generation {generation}: Best={best_strategy.profit_percentage:.2f}%, "
            f"Avg={avg_profit:.2f}%, Diversity={diversity:.2f}, "
            f"MutRate={mutation_rate:.2f}, Time={duration:.2f}s"
        )
    
    def log_strategy_performance(self, strategy, detailed=False):
        """
        Log performance metrics for a strategy.
        
        Args:
            strategy (Strategy): The strategy to log
            detailed (bool): Whether to include detailed metrics
        """
        # Basic performance metrics
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': strategy.grammar_string,
            'generation': strategy.generation,
            'profit_percentage': strategy.profit_percentage,
            'initial_portfolio': strategy.initial_portfolio,
            'final_portfolio': strategy.final_portfolio
        }
        
        # Add detailed metrics if requested
        if detailed and hasattr(strategy, 'metrics'):
            metrics.update(strategy.metrics)
        
        # Add trade summary if available
        if hasattr(strategy, 'trades') and strategy.trades:
            metrics['total_trades'] = len(strategy.trades)
            
            # Calculate win/loss counts
            win_trades = [t for t in strategy.trades if t['profit_pct'] > 0]
            loss_trades = [t for t in strategy.trades if t['profit_pct'] <= 0]
            
            metrics['win_trades'] = len(win_trades)
            metrics['loss_trades'] = len(loss_trades)
            metrics['win_rate'] = len(win_trades) / len(strategy.trades) if strategy.trades else 0
            
            # Average trade metrics
            if win_trades:
                metrics['avg_win'] = sum(t['profit_pct'] for t in win_trades) / len(win_trades)
            if loss_trades:
                metrics['avg_loss'] = sum(t['profit_pct'] for t in loss_trades) / len(loss_trades)
        
        # Log to performance JSON (append)
        try:
            if os.path.exists(self.performance_log_path):
                with open(self.performance_log_path, 'r') as f:
                    performances = json.load(f)
            else:
                performances = []
            
            performances.append(metrics)
            
            with open(self.performance_log_path, 'w') as f:
                json.dump(performances, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error logging strategy performance: {str(e)}")
        
        # Log basic info to system log
        self.logger.info(
            f"Strategy performance logged: {strategy.profit_percentage:.2f}% profit, "
            f"Generation {strategy.generation}"
        )
    
    def log_error(self, error_msg, exception=None):
        """
        Log an error with details.
        
        Args:
            error_msg (str): Error message
            exception (Exception, optional): Exception object if available
        """
        if exception:
            self.logger.error(f"{error_msg}: {str(exception)}")
        else:
            self.logger.error(error_msg)
    
    def log_warning(self, warning_msg):
        """Log a warning message."""
        self.logger.warning(warning_msg)
    
    def log_info(self, info_msg):
        """Log an informational message."""
        self.logger.info(info_msg)
    
    def log_debug(self, debug_msg):
        """Log a debug message."""
        self.logger.debug(debug_msg)
    
    def create_evolution_report(self, output_path=None):
        """
        Create a visual report of the evolution process.
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved report
        """
        try:
            # Read evolution log
            evolution_df = pd.read_csv(self.evolution_log_path)
            
            if len(evolution_df) == 0:
                self.logger.warning("No evolution data available for report")
                return None
            
            # Create output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(self.output_dir, f'evolution_report_{timestamp}.png')
            
            # Set plot style
            sns.set_style('whitegrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
            
            # Plot 1: Profit Percentage (Best and Average)
            axs[0].plot(evolution_df['Generation'], evolution_df['Best Profit'], 
                       'b-', linewidth=2, label='Best Strategy')
            axs[0].plot(evolution_df['Generation'], evolution_df['Avg Profit'], 
                       'r--', linewidth=1.5, label='Population Average')
            axs[0].set_ylabel('Profit %')
            axs[0].set_title('Strategy Performance Evolution')
            axs[0].legend()
            axs[0].grid(True)
            
            # Plot 2: Population Diversity
            axs[1].plot(evolution_df['Generation'], evolution_df['Diversity'], 
                       'g-', linewidth=2)
            axs[1].set_ylabel('Diversity')
            axs[1].set_title('Population Diversity')
            axs[1].grid(True)
            
            # Plot 3: Mutation Rate
            axs[2].plot(evolution_df['Generation'], evolution_df['Mutation Rate'], 
                       'purple', linewidth=2)
            axs[2].set_ylabel('Mutation Rate')
            axs[2].set_title('Adaptive Mutation Rate')
            axs[2].grid(True)
            
            # Plot 4: Generation Duration
            axs[3].bar(evolution_df['Generation'], evolution_df['Duration'], 
                      color='orange', alpha=0.7)
            axs[3].set_xlabel('Generation')
            axs[3].set_ylabel('Time (seconds)')
            axs[3].set_title('Generation Duration')
            axs[3].grid(True)
            
            # Set x-axis ticks to integers
            axs[3].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path)
            plt.close()
            
            self.logger.info(f"Evolution report saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating evolution report: {str(e)}")
            return None
    
    def export_logs(self, format_type='csv'):
        """
        Export logs to various formats.
        
        Args:
            format_type (str): Export format ('csv', 'json', 'html')
            
        Returns:
            str: Path to the exported file
        """
        try:
            # Read evolution log
            evolution_df = pd.read_csv(self.evolution_log_path)
            
            # Create export path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_path = os.path.join(self.output_dir, f'export_{timestamp}.{format_type}')
            
            # Export based on format
            if format_type == 'csv':
                evolution_df.to_csv(export_path, index=False)
            elif format_type == 'json':
                evolution_df.to_json(export_path, orient='records', indent=2)
            elif format_type == 'html':
                # Create a simple HTML report
                html_content = f"""
                <html>
                <head>
                    <title>Trading Strategy Evolution Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333366; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    <h1>Trading Strategy Evolution Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <h2>Evolution Summary</h2>
                    <p>Total Generations: {len(evolution_df)}</p>
                    <p>Best Profit: {evolution_df['Best Profit'].max():.2f}%</p>
                    <p>Average Diversity: {evolution_df['Diversity'].mean():.2f}</p>
                    <h2>Generation Details</h2>
                    {evolution_df.to_html(index=False)}
                </body>
                </html>
                """
                
                with open(export_path, 'w') as f:
                    f.write(html_content)
            else:
                self.logger.error(f"Unsupported export format: {format_type}")
                return None
            
            self.logger.info(f"Logs exported to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Error exporting logs: {str(e)}")
            return None 