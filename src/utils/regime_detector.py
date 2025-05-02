import logging
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"

class MarketRegimeDetector:
    """Detects market regimes based on price action and volatility"""
    
    def __init__(self,
                 lookback_period: int = 20,
                 volatility_threshold: float = 0.02,
                 trend_threshold: float = 0.1):
        """
        Initialize regime detector
        
        Args:
            lookback_period: Period for calculating indicators
            volatility_threshold: Threshold for high/low volatility
            trend_threshold: Threshold for trend strength
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        
        logger.info(f"MarketRegimeDetector initialized with lookback_period={lookback_period}, "
                   f"volatility_threshold={volatility_threshold}, trend_threshold={trend_threshold}")
    
    def detect_regime(self,
                     prices: pd.Series,
                     volume: Optional[pd.Series] = None) -> Tuple[MarketRegime, Dict[MarketRegime, float]]:
        """
        Detect current market regime
        
        Args:
            prices: Price series
            volume: Optional volume series
            
        Returns:
            Tuple of (detected regime, confidence scores)
        """
        try:
            # Calculate indicators
            trend_strength = self._calculate_trend_strength(prices)
            volatility = self._calculate_volatility(prices)
            
            # Calculate regime scores
            scores = {
                MarketRegime.TRENDING_UP: self._calculate_uptrend_score(trend_strength),
                MarketRegime.TRENDING_DOWN: self._calculate_downtrend_score(trend_strength),
                MarketRegime.RANGING: self._calculate_ranging_score(trend_strength),
                MarketRegime.HIGH_VOLATILITY: self._calculate_high_volatility_score(volatility),
                MarketRegime.LOW_VOLATILITY: self._calculate_low_volatility_score(volatility),
                MarketRegime.UNKNOWN: 0.1  # Base score for unknown regime
            }
            
            # Normalize scores
            total = sum(scores.values())
            scores = {k: v/total for k, v in scores.items()}
            
            # Select regime with highest score
            detected_regime = max(scores.items(), key=lambda x: x[1])[0]
            
            return detected_regime, scores
            
        except Exception as e:
            logger.error(f"Regime detection failed: {str(e)}")
            return MarketRegime.UNKNOWN, {r: 0.0 for r in MarketRegime}
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using ADX-like calculation"""
        try:
            # Calculate price changes
            changes = prices.diff()
            
            # Calculate directional movement
            pos_dm = pd.Series(0.0, index=prices.index)
            neg_dm = pd.Series(0.0, index=prices.index)
            
            # Calculate +DM and -DM
            pos_dm.iloc[1:] = np.where(
                changes.iloc[1:] > 0,
                changes.iloc[1:],
                0.0
            )
            neg_dm.iloc[1:] = np.where(
                changes.iloc[1:] < 0,
                -changes.iloc[1:],
                0.0
            )
            
            # Smooth directional movement
            pos_dm = pos_dm.rolling(window=self.lookback_period).mean()
            neg_dm = neg_dm.rolling(window=self.lookback_period).mean()
            
            # Calculate trend strength
            tr = self._calculate_true_range(prices)
            tr_smooth = tr.rolling(window=self.lookback_period).mean()
            
            pos_di = 100 * (pos_dm / tr_smooth)
            neg_di = 100 * (neg_dm / tr_smooth)
            
            # Calculate ADX
            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
            adx = dx.rolling(window=self.lookback_period).mean()
            
            # Return latest ADX value
            return float(adx.iloc[-1]) / 100.0 if not pd.isna(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
    
    def _calculate_true_range(self, prices: pd.Series) -> pd.Series:
        """Calculate true range"""
        high = prices.rolling(window=2).max()
        low = prices.rolling(window=2).min()
        return high - low
    
    def _calculate_volatility(self, prices: pd.Series) -> float:
        """Calculate price volatility"""
        try:
            # Calculate returns
            returns = prices.pct_change()
            
            # Calculate volatility
            volatility = returns.rolling(window=self.lookback_period).std()
            
            # Return latest volatility
            return float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
    
    def _calculate_uptrend_score(self, trend_strength: float) -> float:
        """Calculate uptrend score"""
        if trend_strength > self.trend_threshold:
            return 0.8 * trend_strength
        return 0.2 * trend_strength
    
    def _calculate_downtrend_score(self, trend_strength: float) -> float:
        """Calculate downtrend score"""
        if trend_strength > self.trend_threshold:
            return 0.8 * trend_strength
        return 0.2 * trend_strength
    
    def _calculate_ranging_score(self, trend_strength: float) -> float:
        """Calculate ranging market score"""
        return 1.0 - trend_strength
    
    def _calculate_high_volatility_score(self, volatility: float) -> float:
        """Calculate high volatility score"""
        if volatility > self.volatility_threshold:
            return 0.8 * min(volatility / self.volatility_threshold, 2.0)
        return 0.2 * (volatility / self.volatility_threshold)
    
    def _calculate_low_volatility_score(self, volatility: float) -> float:
        """Calculate low volatility score"""
        if volatility <= self.volatility_threshold:
            return 0.8 * (1.0 - volatility / self.volatility_threshold)
        return 0.2 * (1.0 - min(volatility / self.volatility_threshold, 1.0)) 