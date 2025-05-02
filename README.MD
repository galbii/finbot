# Trading Strategy Evolution System

A sophisticated trading strategy evolution system that uses grammatical evolution to develop and optimize trading strategies for financial markets.

## Project Structure

```
.
├── src/                    # Source code
│   ├── core/              # Core trading components
│   │   ├── bot.py         # Main trading bot implementation
│   │   ├── strategy.py    # Strategy definition and management
│   │   └── data_fetcher.py # Data fetching and management
│   ├── evolution/         # Strategy evolution components
│   │   ├── evolution_manager.py
│   │   └── grammar_parser.py
│   └── run_bot.py         # Main entry point
├── tests/                 # Test suite
├── data/                  # Data storage
├── logs/                  # Log files
├── plots/                 # Generated plots
├── reports/              # Analysis reports
├── .env.example          # Example environment variables
├── requirements.txt      # Python dependencies
├── setup.py             # Package configuration
├── architecture.md      # Technical architecture documentation
└── grammar.md           # Strategy grammar documentation
```

## Key Features

- Grammatical evolution for strategy development
- Advanced backtesting capabilities
- Performance visualization and analysis
- Robust data management using Alpha Vantage API
- Comprehensive logging system
- Pattern analysis and insights generation
- Modern visualization with matplotlib
- Strategy performance metrics calculation

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Alpha Vantage API key
```

## Usage

Run the trading bot:
```bash
python src/run_bot.py
```

The bot will:
1. Fetch historical data
2. Calculate technical indicators
3. Evolve trading strategies
4. Backtest the best strategy
5. Generate performance analysis and visualizations

## Configuration

The bot can be configured through environment variables and command-line arguments:

- `ALPHA_VANTAGE_API_KEY`: API key for data fetching
- `START_DATE`: Start date for backtesting (YYYY-MM-DD)
- `END_DATE`: End date for backtesting (YYYY-MM-DD)
- `POPULATION_SIZE`: Size of the evolution population (default: 50)
- `GENERATIONS`: Number of generations to evolve (default: 20)

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
This project follows PEP 8 guidelines. Use black for formatting:
```bash
black src/
```

## Documentation

For detailed technical documentation, see:
- [Architecture Documentation](architecture.md)
- [Grammar Documentation](grammar.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

