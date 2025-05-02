import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Visualization')

class PerformanceVisualizer:
    """
    Visualization tools for backtesting results.
    
    This class is responsible for:
    - Creating performance charts and plots
    - Generating trade analysis visualizations
    - Exporting results to various formats
    """
    
    def __init__(self, output_dir='plots'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualization outputs
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Set visualization style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
        logger.info("PerformanceVisualizer initialized")
    
    def plot_equity_curve(self, portfolio_history, benchmark=None, title=None, save_path=None):
        """
        Plot equity curve, optionally with benchmark comparison.
        
        Args:
            portfolio_history (pd.DataFrame): Portfolio history from backtest
            benchmark (pd.Series, optional): Benchmark returns for comparison
            title (str, optional): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portfolio value
        portfolio_history['portfolio_value'].plot(ax=ax, linewidth=2, label='Strategy')
        
        # Add benchmark if provided
        if benchmark is not None and len(benchmark) > 0:
            # Reindex benchmark to match portfolio history dates
            benchmark = benchmark.reindex(portfolio_history.index, method='ffill')
            
            # Scale benchmark to start at the same value as portfolio
            initial_value = portfolio_history['portfolio_value'].iloc[0]
            scaled_benchmark = benchmark / benchmark.iloc[0] * initial_value
            
            scaled_benchmark.plot(ax=ax, linewidth=2, linestyle='--', label='Benchmark')
        
        # Format plot
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Equity Curve')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True)
        ax.legend()
        
        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")
        
        return fig
    
    def plot_drawdown_chart(self, portfolio_history, title=None, save_path=None):
        """
        Plot drawdown chart showing periods of decline from peaks.
        
        Args:
            portfolio_history (pd.DataFrame): Portfolio history from backtest
            title (str, optional): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate daily returns
        portfolio_values = portfolio_history['portfolio_value']
        daily_returns = portfolio_values.pct_change().dropna()
        
        # Calculate drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        drawdown.plot(ax=ax, kind='area', color='red', alpha=0.3)
        
        # Format plot
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Drawdown Chart')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Annotate maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd = drawdown.min()
        ax.annotate(f'Max Drawdown: {max_dd:.1%}', 
                 xy=(max_dd_idx, max_dd),
                 xytext=(max_dd_idx, max_dd/2),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Drawdown chart saved to {save_path}")
        
        return fig
    
    def plot_monthly_returns_heatmap(self, portfolio_history, title=None, save_path=None):
        """
        Plot heatmap of monthly returns.
        
        Args:
            portfolio_history (pd.DataFrame): Portfolio history from backtest
            title (str, optional): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate daily returns
        daily_returns = portfolio_history['portfolio_value'].pct_change().dropna()
        
        # Create monthly returns
        monthly_returns = daily_returns.groupby([
            daily_returns.index.year.rename('Year'),
            daily_returns.index.month.rename('Month')
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        # Pivot data for heatmap
        monthly_returns = monthly_returns.unstack('Month')
        
        # Replace month numbers with names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns.columns = [month_names[i-1] for i in monthly_returns.columns]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Monthly Return'})
        
        # Format plot
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Monthly Returns Heatmap')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Monthly returns heatmap saved to {save_path}")
        
        return fig
    
    def plot_trade_analysis(self, trades, title=None, save_path=None):
        """
        Plot trade analysis charts.
        
        Args:
            trades (list): List of trade dictionaries
            title (str, optional): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if not trades:
            logger.warning("No trades to analyze")
            return None
        
        # Extract sell trades for profit analysis
        sell_trades = [t for t in trades if t.get('type') == 'SELL']
        if not sell_trades:
            logger.warning("No sell trades to analyze")
            return None
        
        # Create DataFrame from trades
        trade_df = pd.DataFrame(sell_trades)
        
        # Convert date strings to datetime if necessary
        if isinstance(trade_df['date'].iloc[0], str):
            trade_df['date'] = pd.to_datetime(trade_df['date'])
        
        # Add trade duration
        if 'entry_date' in trade_df.columns:
            trade_df['duration'] = (trade_df['date'] - trade_df['entry_date']).dt.days
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Profit per trade
        trade_df['trade_number'] = range(1, len(trade_df) + 1)
        axes[0, 0].bar(trade_df['trade_number'], trade_df['profit'], 
                     color=trade_df['profit'].apply(lambda x: 'green' if x > 0 else 'red'))
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 0].set_title('Profit per Trade')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Profit ($)')
        axes[0, 0].grid(True, axis='y')
        
        # Plot 2: Profit distribution
        profits = trade_df['profit'].values
        axes[0, 1].hist(profits, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].set_title('Profit Distribution')
        axes[0, 1].set_xlabel('Profit ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Plot 3: Cumulative profit
        cum_profit = np.cumsum(profits)
        axes[1, 0].plot(range(1, len(cum_profit) + 1), cum_profit, marker='o', linestyle='-')
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Cumulative Profit')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative Profit ($)')
        axes[1, 0].grid(True)
        
        # Plot 4: Win/Loss statistics
        win_count = sum(1 for p in profits if p > 0)
        loss_count = sum(1 for p in profits if p <= 0)
        win_rate = win_count / len(profits) if len(profits) > 0 else 0
        
        labels = ['Winning Trades', 'Losing Trades']
        sizes = [win_count, loss_count]
        colors = ['green', 'red']
        explode = (0.1, 0)
        
        axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                     shadow=True, startangle=90)
        axes[1, 1].axis('equal')
        axes[1, 1].set_title(f'Win/Loss Ratio: {win_rate:.2%}')
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=16)
            plt.subplots_adjust(top=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Trade analysis saved to {save_path}")
        
        return fig
    
    def generate_performance_report(self, backtest_results, strategy_name=None, save_path=None):
        """
        Generate a comprehensive performance report.
        
        Args:
            backtest_results (dict): Results from backtest
            strategy_name (str, optional): Name of the strategy
            save_path (str, optional): Path to save the report
            
        Returns:
            str: Path to saved report
        """
        # Extract data from backtest results
        portfolio_history = backtest_results['portfolio_history']
        trades = backtest_results['trades']
        metrics = backtest_results['metrics']
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, f'report_{timestamp}')
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        # Generate plots
        equity_path = os.path.join(report_dir, 'equity_curve.png')
        self.plot_equity_curve(portfolio_history, title=f'Equity Curve: {strategy_name}', save_path=equity_path)
        
        drawdown_path = os.path.join(report_dir, 'drawdown.png')
        self.plot_drawdown_chart(portfolio_history, title=f'Drawdown: {strategy_name}', save_path=drawdown_path)
        
        if len(portfolio_history) > 20:  # Need enough data for monthly returns
            monthly_path = os.path.join(report_dir, 'monthly_returns.png')
            self.plot_monthly_returns_heatmap(portfolio_history, title=f'Monthly Returns: {strategy_name}', save_path=monthly_path)
        
        if trades:
            trade_path = os.path.join(report_dir, 'trade_analysis.png')
            self.plot_trade_analysis(trades, title=f'Trade Analysis: {strategy_name}', save_path=trade_path)
        
        # Generate summary report in HTML format
        html_path = os.path.join(report_dir, 'performance_report.html')
        
        with open(html_path, 'w') as f:
            f.write(f'''
            <html>
            <head>
                <title>Performance Report: {strategy_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333366; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .metrics {{ display: flex; justify-content: space-between; }}
                    .metric-box {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; margin: 10px; flex: 1; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Performance Report: {strategy_name if strategy_name else "Trading Strategy"}</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Performance Metrics</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Returns</h3>
                        <p>Profit: <span class="{('positive' if metrics['profit_percentage'] > 0 else 'negative')}">{metrics['profit_percentage']:.2f}%</span></p>
                        <p>Sharpe Ratio: {metrics['sharpe_ratio']:.2f}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Risk</h3>
                        <p>Max Drawdown: <span class="negative">{metrics['max_drawdown']:.2f}%</span></p>
                    </div>
                    <div class="metric-box">
                        <h3>Trading</h3>
                        <p>Number of Trades: {metrics['num_trades']}</p>
                        <p>Win Rate: {metrics['win_rate']:.2f}%</p>
                        <p>Profit Factor: {metrics['profit_factor']:.2f}</p>
                    </div>
                </div>
                
                <h2>Equity Curve</h2>
                <img src="equity_curve.png" alt="Equity Curve">
                
                <h2>Drawdown Chart</h2>
                <img src="drawdown.png" alt="Drawdown Chart">
            ''')
            
            # Add monthly returns if available
            if len(portfolio_history) > 20:
                f.write('''
                <h2>Monthly Returns</h2>
                <img src="monthly_returns.png" alt="Monthly Returns">
                ''')
            
            # Add trade analysis if available
            if trades:
                f.write('''
                <h2>Trade Analysis</h2>
                <img src="trade_analysis.png" alt="Trade Analysis">
                
                <h2>Trade List</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Price</th>
                        <th>Shares</th>
                        <th>Value</th>
                        <th>Profit</th>
                    </tr>
                ''')
                
                # Add trade rows
                for trade in trades:
                    trade_date = trade['date']
                    if isinstance(trade_date, str):
                        trade_date_str = trade_date
                    else:
                        trade_date_str = trade_date.strftime('%Y-%m-%d')
                    
                    profit_class = ""
                    profit_str = ""
                    if trade.get('type') == 'SELL' and 'profit' in trade:
                        profit_class = 'positive' if trade['profit'] > 0 else 'negative'
                        profit_str = f"<span class='{profit_class}'>${trade['profit']:.2f}</span>"
                    
                    f.write(f'''
                    <tr>
                        <td>{trade_date_str}</td>
                        <td>{trade.get('type', '')}</td>
                        <td>${trade.get('price', 0):.2f}</td>
                        <td>{trade.get('shares', 0):.2f}</td>
                        <td>${trade.get('value', 0):.2f}</td>
                        <td>{profit_str}</td>
                    </tr>
                    ''')
                
                f.write('</table>')
            
            f.write('''
            </body>
            </html>
            ''')
        
        logger.info(f"Performance report generated at {html_path}")
        return report_dir 