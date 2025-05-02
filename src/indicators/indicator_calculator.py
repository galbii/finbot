import pandas as pd
import numpy as np
import logging
from functools import lru_cache
import ta
import warnings
from typing import Dict, Any, Union, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IndicatorCalculator')

class IndicatorCalculator:
    """
    Handles calculation and caching of technical indicators.
    
    This class is responsible for:
    - Calculating technical indicators based on OHLCV data
    - Caching results for efficient reuse
    - Providing accessor methods for each indicator
    - Supporting incremental updates for performance
    """
    
    def __init__(self):
        """Initialize the IndicatorCalculator"""
        self.indicator_cache = {}
        self.incremental_data = {}  # Store incremental data for indicators
        self.last_calculated_indices = {}  # Track the last calculated index for each indicator
        logger.info("IndicatorCalculator initialized")
    
    def clear_cache(self):
        """Clear the indicator cache"""
        self.indicator_cache = {}
        self.incremental_data = {}
        self.last_calculated_indices = {}
        logger.info("Indicator cache cleared")
    
    def _get_cache_key(self, indicator_name: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for an indicator.
        
        Args:
            indicator_name (str): Name of the indicator
            params (dict): Parameters for the indicator
            
        Returns:
            str: Cache key
        """
        params_str = '_'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return f"{indicator_name}_{params_str}"
    
    def calculate_indicator(self, df: pd.DataFrame, indicator_name: str, **params) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate a technical indicator and cache the result.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            indicator_name (str): Name of the indicator to calculate
            **params: Parameters for the indicator
            
        Returns:
            pd.Series or pd.DataFrame: Calculated indicator
        """
        # Generate cache key
        cache_key = self._get_cache_key(indicator_name, params)
        
        # Check if we can perform an incremental update
        if cache_key in self.indicator_cache:
            # Get existing result
            existing_result = self.indicator_cache[cache_key]
            last_idx = self.last_calculated_indices.get(cache_key, 0)
            
            # Check if we need to update (only if df has more rows than what we've already calculated)
            if len(df) > last_idx and isinstance(existing_result, (pd.Series, pd.DataFrame)):
                try:
                    # Perform incremental update only on new data
                    updated_result = self._incremental_update(df, indicator_name, existing_result, last_idx, **params)
                    if updated_result is not None:
                        self.indicator_cache[cache_key] = updated_result
                        self.last_calculated_indices[cache_key] = len(df)
                        logger.debug(f"Incrementally updated {cache_key} from index {last_idx} to {len(df)}")
                        return updated_result
                except Exception as e:
                    logger.warning(f"Failed to perform incremental update for {indicator_name}: {e}. Recalculating.")
                    # Fall back to full recalculation
            else:
                logger.debug(f"Using cached result for {cache_key}")
                return existing_result
        
        # Calculate the indicator
        try:
            result = self._calculate_indicator_impl(df, indicator_name, **params)
            
            # Cache the result
            self.indicator_cache[cache_key] = result
            self.last_calculated_indices[cache_key] = len(df)
            logger.debug(f"Calculated and cached {cache_key}")
            
            return result
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {e}")
            return None
    
    def _incremental_update(self, df: pd.DataFrame, indicator_name: str, 
                           existing_result: Union[pd.Series, pd.DataFrame], 
                           last_idx: int, **params) -> Union[pd.Series, pd.DataFrame, None]:
        """
        Incrementally update an indicator with new data points.
        
        Args:
            df (pd.DataFrame): Complete DataFrame with OHLCV data
            indicator_name (str): Name of the indicator to calculate
            existing_result: Previously calculated result
            last_idx: Last calculated index
            **params: Parameters for the indicator
            
        Returns:
            Updated indicator values or None if incremental update is not supported
        """
        indicator_name = indicator_name.upper()
        
        # Some indicators can be updated incrementally, others need full recalculation
        if indicator_name == 'RSI':
            period = params.get('period', 14)
            # For RSI we need to keep track of avg_gain and avg_loss
            if last_idx == 0:
                # No incremental data yet, can't update incrementally
                return None
            
            # Get incremental data for this indicator if available
            incr_key = self._get_cache_key(indicator_name, params)
            incremental_data = self.incremental_data.get(incr_key)
            
            if incremental_data is None:
                # Initialize incremental data if not available
                if last_idx < period + 1:
                    # Not enough data for incremental update
                    return None
                
                # Calculate the last gain/loss values
                close_series = df['Close'].iloc[:last_idx]
                delta = close_series.diff(1).iloc[1:]
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=period).mean().iloc[-1]
                avg_loss = loss.rolling(window=period).mean().iloc[-1]
                
                incremental_data = {
                    'avg_gain': avg_gain,
                    'avg_loss': avg_loss
                }
                self.incremental_data[incr_key] = incremental_data
            
            # Get new data points
            new_data = df.iloc[last_idx:]
            if len(new_data) == 0:
                return existing_result
            
            # Create copy of existing result to avoid modifying the original
            result = existing_result.copy()
            
            # Calculate new RSI values incrementally
            avg_gain = incremental_data['avg_gain']
            avg_loss = incremental_data['avg_loss']
            
            close_series = df['Close']
            
            for idx in range(last_idx, len(df)):
                if idx == 0:
                    continue
                
                # Calculate current gain/loss
                current_close = close_series.iloc[idx]
                prev_close = close_series.iloc[idx-1]
                delta = current_close - prev_close
                
                current_gain = max(delta, 0)
                current_loss = max(-delta, 0)
                
                # Update average gain/loss
                avg_gain = ((period - 1) * avg_gain + current_gain) / period
                avg_loss = ((period - 1) * avg_loss + current_loss) / period
                
                # Calculate RSI
                rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
                rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100
                
                # Update result
                result.iloc[idx] = rsi
            
            # Update incremental data
            incremental_data['avg_gain'] = avg_gain
            incremental_data['avg_loss'] = avg_loss
            self.incremental_data[incr_key] = incremental_data
            
            return result
            
        elif indicator_name == 'SMA':
            # SMA can be updated incrementally using a rolling window approach
            period = params.get('period', 20)
            
            # If we don't have at least one full window, fall back to full recalculation
            if last_idx < period:
                return None
                
            # Get new data points
            new_data = df.iloc[last_idx:]
            if len(new_data) == 0:
                return existing_result
                
            # Create copy of existing result to avoid modifying the original
            result = existing_result.copy()
            
            # Calculate new SMA values incrementally using vectorized operations
            for i in range(last_idx, len(df)):
                window_start = max(0, i - period + 1)
                result.iloc[i] = df['Close'].iloc[window_start:i+1].mean()
                
            return result
            
        # For other indicators, indicate that incremental update is not supported
        return None
    
    def _calculate_indicator_impl(self, df: pd.DataFrame, indicator_name: str, **params) -> Union[pd.Series, pd.DataFrame]:
        """
        Implementation of indicator calculations using vectorized operations.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            indicator_name (str): Name of the indicator to calculate
            **params: Parameters for the indicator
            
        Returns:
            pd.Series or pd.DataFrame: Calculated indicator
        """
        # Convert column names to match ta-lib expectations
        df_copy = df.copy()
        if 'Close' in df.columns and 'close' not in df.columns:
            df_copy['close'] = df['Close']
        if 'Open' in df.columns and 'open' not in df.columns:
            df_copy['open'] = df['Open']
        if 'High' in df.columns and 'high' not in df.columns:
            df_copy['high'] = df['High']
        if 'Low' in df.columns and 'low' not in df.columns:
            df_copy['low'] = df['Low']
        if 'Volume' in df.columns and 'volume' not in df.columns:
            df_copy['volume'] = df['Volume']
            
        indicator_name = indicator_name.upper()
        
        if indicator_name == 'RSI':
            period = params.get('period', 14)
            logger.debug(f"Calculating RSI with period={period} using vectorized method")
            
            # Calculate RSI in a vectorized way
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rsi = ta.momentum.RSIIndicator(df_copy['close'], window=period)
                return rsi.rsi()
        
        elif indicator_name == 'MACD':
            fast_period = params.get('fast_period', 12)
            slow_period = params.get('slow_period', 26)
            signal_period = params.get('signal_period', 9)
            logger.debug(f"Calculating MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                macd = ta.trend.MACD(
                    df_copy['close'], 
                    window_fast=fast_period, 
                    window_slow=slow_period, 
                    window_sign=signal_period
                )
                
                result = pd.DataFrame({
                    'macd': macd.macd(),
                    'signal': macd.macd_signal(),
                    'histogram': macd.macd_diff()
                })
                
                return result
        
        elif indicator_name == 'BB' or indicator_name == 'BOLLINGER_BANDS':
            window = params.get('window', 20)
            std = params.get('std', 2)
            logger.debug(f"Calculating Bollinger Bands with window={window}, std={std}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bb = ta.volatility.BollingerBands(
                    df_copy['close'], 
                    window=window, 
                    window_dev=std
                )
                
                result = pd.DataFrame({
                    'upper': bb.bollinger_hband(),
                    'middle': bb.bollinger_mavg(),
                    'lower': bb.bollinger_lband()
                })
                
                return result
        
        elif indicator_name == 'SMA':
            period = params.get('period', 20)
            logger.debug(f"Calculating SMA with period={period}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ta.trend.SMAIndicator(df_copy['close'], window=period).sma_indicator()
        
        elif indicator_name == 'EMA':
            period = params.get('period', 20)
            logger.debug(f"Calculating EMA with period={period}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ta.trend.EMAIndicator(df_copy['close'], window=period).ema_indicator()
        
        else:
            raise ValueError(f"Unsupported indicator: {indicator_name}")
    
    @lru_cache(maxsize=128)
    def detect_crossover(self, series1_id: int, series2_id: int) -> pd.Series:
        """
        Detect crossovers between two series.
        Modified to use series IDs for caching purposes.
        
        Args:
            series1_id: ID of first series (use id(series) when calling)
            series2_id: ID of second series (use id(series) when calling)
            
        Returns:
            pd.Series: Boolean series with True at crossover points where series1 crosses above series2
        """
        # Convert IDs back to series
        # This is just to allow the function to be cached with lru_cache
        # In actual use, pass id(series1) and id(series2) as arguments
        for obj_id, obj in list(globals().items()):
            if id(obj) == series1_id:
                series1 = obj
            if id(obj) == series2_id:
                series2 = obj
        
        # Previous day's comparison (True if series1 was below series2)
        prev_below = (series1.shift(1) < series2.shift(1))
        
        # Current day's comparison (True if series1 is above series2)
        curr_above = (series1 > series2)
        
        # Crossover occurs when previous was below and current is above
        crossover = prev_below & curr_above
        
        # First value will be NaN due to shift, set it to False
        crossover.iloc[0] = False
        
        return crossover
    
    @lru_cache(maxsize=128)
    def detect_crossunder(self, series1_id: int, series2_id: int) -> pd.Series:
        """
        Detect crossunders between two series.
        Modified to use series IDs for caching purposes.
        
        Args:
            series1_id: ID of first series (use id(series) when calling)
            series2_id: ID of second series (use id(series) when calling)
            
        Returns:
            pd.Series: Boolean series with True at crossunder points where series1 crosses below series2
        """
        # Convert IDs back to series
        for obj_id, obj in list(globals().items()):
            if id(obj) == series1_id:
                series1 = obj
            if id(obj) == series2_id:
                series2 = obj
                
        # Previous day's comparison (True if series1 was above series2)
        prev_above = (series1.shift(1) > series2.shift(1))
        
        # Current day's comparison (True if series1 is below series2)
        curr_below = (series1 < series2)
        
        # Crossunder occurs when previous was above and current is below
        crossunder = prev_above & curr_below
        
        # First value will be NaN due to shift, set it to False
        crossunder.iloc[0] = False
        
        return crossunder 
    
    def batch_calculate(self, df: pd.DataFrame, indicators_config: list) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Calculate multiple indicators in a single batch operation.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            indicators_config: List of indicator configurations
                               e.g. [{'name': 'RSI', 'params': {'period': 14}}, 
                                     {'name': 'SMA', 'params': {'period': 20}}]
            
        Returns:
            dict: Dictionary of indicator name to calculated values
        """
        results = {}
        
        for indicator_config in indicators_config:
            name = indicator_config['name']
            params = indicator_config.get('params', {})
            key = self._get_cache_key(name, params)
            
            result = self.calculate_indicator(df, name, **params)
            results[key] = result
            
        return results 