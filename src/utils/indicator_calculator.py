class IndicatorCalculator:
    """
    Calculator for technical indicators used in trading strategies.
    """
    
    def __init__(self, data):
        """
        Initialize the calculator with price data.
        
        Args:
            data (pd.DataFrame): Price data with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        """
        self.data = data
        self._cache = {}
    
    def calculate_rsi(self, period, date):
        """
        Calculate RSI for a given date.
        
        Args:
            period (int): RSI period
            date (str): Date to calculate RSI for
            
        Returns:
            float: RSI value
        """
        cache_key = f'rsi_{period}_{date}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get data up to the given date
        data_up_to_date = self.data[self.data['date'] <= date]
        if len(data_up_to_date) < period:
            return None
        
        # Calculate RSI
        delta = data_up_to_date['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        result = rsi.iloc[-1]
        self._cache[cache_key] = result
        return result
    
    def calculate_macd(self, fast_period, slow_period, signal_period, date):
        """
        Calculate MACD and signal line for a given date.
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            date (str): Date to calculate MACD for
            
        Returns:
            tuple: (MACD line, Signal line)
        """
        cache_key = f'macd_{fast_period}_{slow_period}_{signal_period}_{date}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get data up to the given date
        data_up_to_date = self.data[self.data['date'] <= date]
        if len(data_up_to_date) < slow_period + signal_period:
            return None, None
        
        # Calculate MACD
        fast_ema = data_up_to_date['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data_up_to_date['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        result = (macd_line.iloc[-1], signal_line.iloc[-1])
        self._cache[cache_key] = result
        return result
    
    def calculate_bollinger_bands(self, window, std, date):
        """
        Calculate Bollinger Bands for a given date.
        
        Args:
            window (int): Moving average window
            std (float): Number of standard deviations
            date (str): Date to calculate bands for
            
        Returns:
            tuple: (Upper band, Lower band)
        """
        cache_key = f'bb_{window}_{std}_{date}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get data up to the given date
        data_up_to_date = self.data[self.data['date'] <= date]
        if len(data_up_to_date) < window:
            return None, None
        
        # Calculate Bollinger Bands
        ma = data_up_to_date['close'].rolling(window=window).mean()
        std_dev = data_up_to_date['close'].rolling(window=window).std()
        upper_band = ma + (std_dev * std)
        lower_band = ma - (std_dev * std)
        
        result = (upper_band.iloc[-1], lower_band.iloc[-1])
        self._cache[cache_key] = result
        return result
    
    def calculate_ma(self, ma_type, period, date):
        """
        Calculate moving average for a given date.
        
        Args:
            ma_type (str): Type of moving average ('SMA' or 'EMA')
            period (int): Moving average period
            date (str): Date to calculate MA for
            
        Returns:
            float: Moving average value
        """
        cache_key = f'ma_{ma_type}_{period}_{date}'
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get data up to the given date
        data_up_to_date = self.data[self.data['date'] <= date]
        if len(data_up_to_date) < period:
            return None
        
        # Calculate moving average
        if ma_type == 'SMA':
            ma = data_up_to_date['close'].rolling(window=period).mean()
        elif ma_type == 'EMA':
            ma = data_up_to_date['close'].ewm(span=period, adjust=False).mean()
        else:
            raise ValueError(f"Unknown moving average type: {ma_type}")
        
        result = ma.iloc[-1]
        self._cache[cache_key] = result
        return result
    
    def get_price(self, date):
        """
        Get the closing price for a given date.
        
        Args:
            date (str): Date to get price for
            
        Returns:
            float: Closing price
        """
        price = self.data[self.data['date'] == date]['close'].iloc[0]
        return price
    
    def crosses_above(self, series1, series2, date):
        """
        Check if series1 crosses above series2 at the given date.
        
        Args:
            series1 (float): First series value
            series2 (float): Second series value
            date (str): Date to check
            
        Returns:
            bool: True if series1 crosses above series2
        """
        if series1 is None or series2 is None:
            return False
        
        # Get previous date
        date_idx = self.data[self.data['date'] <= date].index[-1]
        if date_idx == 0:
            return False
        
        prev_date = self.data.index[date_idx - 1]
        
        # Get previous values
        if isinstance(series1, (int, float)):
            prev_series1 = series1
        else:
            prev_series1 = self._get_previous_value(series1, prev_date)
        
        if isinstance(series2, (int, float)):
            prev_series2 = series2
        else:
            prev_series2 = self._get_previous_value(series2, prev_date)
        
        if prev_series1 is None or prev_series2 is None:
            return False
        
        return prev_series1 <= prev_series2 and series1 > series2
    
    def crosses_below(self, series1, series2, date):
        """
        Check if series1 crosses below series2 at the given date.
        
        Args:
            series1 (float): First series value
            series2 (float): Second series value
            date (str): Date to check
            
        Returns:
            bool: True if series1 crosses below series2
        """
        if series1 is None or series2 is None:
            return False
        
        # Get previous date
        date_idx = self.data[self.data['date'] <= date].index[-1]
        if date_idx == 0:
            return False
        
        prev_date = self.data.index[date_idx - 1]
        
        # Get previous values
        if isinstance(series1, (int, float)):
            prev_series1 = series1
        else:
            prev_series1 = self._get_previous_value(series1, prev_date)
        
        if isinstance(series2, (int, float)):
            prev_series2 = series2
        else:
            prev_series2 = self._get_previous_value(series2, prev_date)
        
        if prev_series1 is None or prev_series2 is None:
            return False
        
        return prev_series1 >= prev_series2 and series1 < series2
    
    def _get_previous_value(self, indicator_func, date):
        """
        Get the previous value of an indicator.
        
        Args:
            indicator_func (function): Function to calculate indicator
            date (str): Date to get previous value for
            
        Returns:
            float: Previous indicator value
        """
        # This is a simplified version - in practice, you'd need to handle
        # different types of indicators and their parameters
        try:
            return indicator_func(date)
        except:
            return None 