# Trading Strategy Grammar for Grammatical Evolution

This grammar defines a structure for evolving trading strategies based on technical indicators
like RSI, MACD, Bollinger Bands, etc. The grammar is designed to be:
1. Extensible - easily add new indicators
2. Composable - combine conditions with logical operators
3. Machine learning friendly - structured to facilitate evaluation

## Grammar Structure

The grammar is defined in Backus-Naur Form (BNF) and supports the following components:

1. **Entry and Exit Rules**
   - Simple conditions: `BUY WHEN RSI(14) < 30`
   - Compound conditions: `BUY WHEN RSI(14) < 30 AND MACD(12,26,9) CROSSES ABOVE SIGNAL`
   - Multiple conditions: `BUY WHEN RSI(14) < 30 OR PRICE < LOWER_BB(20,2.0)`

2. **Technical Indicators**
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Moving Averages (SMA, EMA)
   - Price Action
   - Volume Analysis

3. **Logical Operators**
   - AND
   - OR
   - NOT

4. **Position Sizing**
   - Fixed amount
   - Percentage of portfolio
   - Risk-based sizing

## Grammar Definition

```python
GRAMMAR = {
    # Starting rule
    'strategy': [
        '<entry_rules> <exit_rules>',
        '<entry_rules> <exit_rules> <position_sizing>',
    ],
    
    # Entry and exit rules
    'entry_rules': [
        'BUY WHEN <condition>',
        'BUY WHEN <condition> AND <condition>',
        'BUY WHEN <condition> OR <condition>',
    ],
    
    'exit_rules': [
        'SELL WHEN <condition>',
        'SELL WHEN <condition> AND <condition>',
        'SELL WHEN <condition> OR <condition>',
    ],
    
    # Optional position sizing
    'position_sizing': [
        'SIZE = <position_size>',
    ],

    'position_size': [
        '<fixed_amount>',
        '<percentage>',
        '<risk_based>',
    ],
    
    'fixed_amount': ['100', '200', '500', '1000'],
    'percentage': ['10%', '25%', '50%', '75%', '100%'],
    'risk_based': ['1% OF PORTFOLIO', '2% OF PORTFOLIO', '5% OF PORTFOLIO'],
    
    # Conditions based on technical indicators
    'condition': [
        # RSI conditions
        '<rsi_condition>',
        # MACD conditions
        '<macd_condition>',
        # Bollinger Bands conditions
        '<bb_condition>',
        # Moving Average conditions
        '<ma_condition>',
        # Price action conditions
        '<price_condition>',
        # Volume conditions
        '<volume_condition>',
        # Compound conditions
        '(<condition> AND <condition>)',
        '(<condition> OR <condition>)',
        'NOT <condition>',
    ],
    
    # RSI specific conditions
    'rsi_condition': [
        'RSI(<period>) < <rsi_value>',
        'RSI(<period>) > <rsi_value>',
        'RSI(<period>) CROSSES ABOVE <rsi_value>',
        'RSI(<period>) CROSSES BELOW <rsi_value>',
    ],
    'period': ['7', '14', '21', '30'],
    'rsi_value': ['20', '30', '40', '50', '60', '70', '80'],
    
    # MACD specific conditions
    'macd_condition': [
        'MACD(<fast>, <slow>, <signal>) CROSSES ABOVE SIGNAL',
        'MACD(<fast>, <slow>, <signal>) CROSSES BELOW SIGNAL',
        'MACD(<fast>, <slow>, <signal>) > 0',
        'MACD(<fast>, <slow>, <signal>) < 0',
        'MACD(<fast>, <slow>, <signal>) CROSSES ABOVE 0',
        'MACD(<fast>, <slow>, <signal>) CROSSES BELOW 0',
    ],
    'fast': ['8', '12', '16'],
    'slow': ['21', '26', '32'],
    'signal': ['5', '9', '13'],
    
    # Bollinger Bands specific conditions
    'bb_condition': [
        'PRICE CROSSES ABOVE UPPER_BB(<bb_period>, <bb_std>)',
        'PRICE CROSSES BELOW LOWER_BB(<bb_period>, <bb_std>)',
        'PRICE > UPPER_BB(<bb_period>, <bb_std>)',
        'PRICE < LOWER_BB(<bb_period>, <bb_std>)',
    ],
    'bb_period': ['14', '20', '30'],
    'bb_std': ['1.5', '2.0', '2.5', '3.0'],
    
    # Moving Average conditions
    'ma_condition': [
        'SMA(<ma_short>) CROSSES ABOVE SMA(<ma_long>)',
        'SMA(<ma_short>) CROSSES BELOW SMA(<ma_long>)',
        'EMA(<ma_short>) CROSSES ABOVE EMA(<ma_long>)',
        'EMA(<ma_short>) CROSSES BELOW EMA(<ma_long>)',
        'PRICE CROSSES ABOVE SMA(<ma_period>)',
        'PRICE CROSSES BELOW SMA(<ma_period>)',
        'PRICE CROSSES ABOVE EMA(<ma_period>)',
        'PRICE CROSSES BELOW EMA(<ma_period>)',
    ],
    'ma_short': ['5', '10', '20', '50'],
    'ma_long': ['50', '100', '200'],
    'ma_period': ['10', '20', '50', '100', '200'],
    
    # Price action conditions
    'price_condition': [
        'PRICE > YESTERDAY_CLOSE',
        'PRICE < YESTERDAY_CLOSE',
        'PRICE > PRICE(<lookback>)',
        'PRICE < PRICE(<lookback>)',
        'PRICE MAKES <bar_count> DAY HIGH',
        'PRICE MAKES <bar_count> DAY LOW',
    ],
    'lookback': ['1', '2', '3', '5', '10'],
    'bar_count': ['5', '10', '20', '50', '100'],
    
    # Volume conditions
    'volume_condition': [
        'VOLUME > SMA(VOLUME, <vol_period>)',
        'VOLUME < SMA(VOLUME, <vol_period>)',
        'VOLUME > <vol_mult> * YESTERDAY_VOLUME',
    ],
    'vol_period': ['5', '10', '20', '50'],
    'vol_mult': ['1.5', '2.0', '3.0'],
}
```

## Example Strategies

1. **Simple RSI Strategy**
   ```
   BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70
   ```

2. **RSI with Position Sizing**
   ```
   BUY WHEN RSI(14) < 30 SELL WHEN RSI(14) > 70 SIZE = 2% OF PORTFOLIO
   ```

3. **Combined RSI and MACD Strategy**
   ```
   BUY WHEN RSI(14) < 30 AND MACD(12, 26, 9) CROSSES ABOVE SIGNAL 
   SELL WHEN RSI(14) > 70 OR MACD(12, 26, 9) CROSSES BELOW SIGNAL
   ```

4. **Bollinger Bands Strategy with Moving Averages**
   ```
   BUY WHEN PRICE < LOWER_BB(20, 2.0) AND SMA(50) > SMA(200) 
   SELL WHEN PRICE > UPPER_BB(20, 2.0) OR SMA(50) CROSSES BELOW SMA(200)
   ```

## Pattern Analysis Examples

The system analyzes patterns in evolved strategies to provide insights:

1. **Market Condition Analysis**
   ```
   Strategy: BUY WHEN RSI(14) < 30
   Analysis:
   - Average RSI at entry: 28.5
   - Standard deviation: 2.1
   - Optimal range: 25-32
   ```

2. **Trade Timing Analysis**
   ```
   Strategy: BUY WHEN MACD(12,26,9) CROSSES ABOVE SIGNAL
   Analysis:
   - Average trade duration: 5.2 days
   - Most profitable duration: 3-7 days
   - Early exit threshold: 10% loss
   ```

3. **Risk Management Analysis**
   ```
   Strategy: BUY WHEN RSI(14) < 30 AND PRICE < LOWER_BB(20,2.0)
   Analysis:
   - Win rate: 65%
   - Average win: 8.2%
   - Average loss: 4.5%
   - Profit factor: 2.1
   ```

4. **Combined Strategy Analysis**
   ```
   Strategy: BUY WHEN RSI(14) < 30 AND MACD(12,26,9) CROSSES ABOVE SIGNAL
   Analysis:
   - Market conditions: Oversold with momentum shift
   - Optimal entry: RSI < 30 and MACD crossover
   - Exit timing: RSI > 60 or MACD crossover
   - Risk management: 2% position sizing
   ```

## Extending the Grammar

To add new indicators or conditions:

1. Define new condition types in the grammar
2. Add corresponding evaluation functions
3. Update the parser to handle new syntax
4. Add pattern analysis for new indicators

Example of adding Stochastic Oscillator:

```python
def extend_grammar_with_stochastic():
    """Add Stochastic oscillator conditions to the grammar"""
    GRAMMAR['condition'].append('<stoch_condition>')
    GRAMMAR['stoch_condition'] = [
        'STOCH_K(<stoch_k_period>, <stoch_d_period>) CROSSES ABOVE <stoch_threshold>',
        'STOCH_K(<stoch_k_period>, <stoch_d_period>) CROSSES BELOW <stoch_threshold>',
        'STOCH_K(<stoch_k_period>, <stoch_d_period>) CROSSES ABOVE STOCH_D',
        'STOCH_K(<stoch_k_period>, <stoch_d_period>) CROSSES BELOW STOCH_D',
    ]
    GRAMMAR['stoch_k_period'] = ['5', '9', '14']
    GRAMMAR['stoch_d_period'] = ['3', '5', '9']
    GRAMMAR['stoch_threshold'] = ['20', '30', '50', '70', '80']
```

## Best Practices

1. **Strategy Design**
   - Combine multiple indicators for robust signals
   - Include both entry and exit conditions
   - Consider position sizing for risk management
   - Test strategies across different market conditions

2. **Pattern Analysis**
   - Analyze market conditions at entry/exit points
   - Track trade duration and timing
   - Monitor risk management metrics
   - Generate actionable insights

3. **Performance Optimization**
   - Use efficient indicator calculations
   - Cache intermediate results
   - Implement early termination for poor strategies
   - Monitor resource usage

4. **Maintenance**
   - Keep grammar documentation up to date
   - Document pattern analysis results
   - Track strategy performance over time
   - Update analysis methods as needed
