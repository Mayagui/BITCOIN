import pandas as pd
import numpy as np
import ta
<<<<<<< HEAD
import logging
from typing import List, Optional

class EnhancedTechnicalIndicators:
    def __init__(
        self,
        data: pd.DataFrame,
        sma_periods: Optional[List[int]] = None,
        ema_periods: Optional[List[int]] = None,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        stoch_k: int = 14,
        stoch_d: int = 3,
        williams_window: int = 14,
        bb_window: int = 20,
        bb_std: int = 2,
        atr_window: int = 14,
        adx_window: int = 14,
        support_resistance_window: int = 20
    ):
        self.data = data
        self.sma_periods = sma_periods or [20, 50, 100, 200]
        self.ema_periods = ema_periods or [20, 50, 100, 200]
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.williams_window = williams_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.atr_window = atr_window
        self.adx_window = adx_window
        self.support_resistance_window = support_resistance_window
        self.logger = logging.getLogger(__name__)

    def add_all_indicators(self) -> pd.DataFrame:
        """Ajoute tous les indicateurs techniques au DataFrame."""
        try:
            self.add_moving_averages()
            self.add_macd()
            self.add_adx()
            self.add_rsi()
            self.add_stochastic()
            self.add_williams_r()
            self.add_bollinger_bands()
            self.add_atr()
            self.add_volume_indicators()
            self.add_support_resistance()
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout des indicateurs : {e}")
        return self.data

    def add_moving_averages(self):
        """Ajoute plusieurs moyennes mobiles (SMA et EMA)."""
        if 'close' not in self.data.columns:
            raise ValueError("Colonne 'close' manquante pour les moyennes mobiles.")
        for period in self.sma_periods:
            self.data[f'SMA_{period}'] = ta.trend.sma_indicator(self.data['close'], window=period)
        for period in self.ema_periods:
            self.data[f'EMA_{period}'] = ta.trend.ema_indicator(self.data['close'], window=period)

    def add_macd(self):
        """Ajoute MACD, signal et histogramme."""
        if 'close' not in self.data.columns:
            raise ValueError("Colonne 'close' manquante pour le MACD.")
        self.data['MACD'] = ta.trend.macd(self.data['close'], window_slow=self.macd_slow, window_fast=self.macd_fast)
        self.data['MACD_Signal'] = ta.trend.macd_signal(self.data['close'], window_slow=self.macd_slow, window_fast=self.macd_fast, window_sign=self.macd_signal)
        self.data['MACD_Hist'] = ta.trend.macd_diff(self.data['close'], window_slow=self.macd_slow, window_fast=self.macd_fast, window_sign=self.macd_signal)

    def add_rsi(self):
        """Ajoute RSI."""
        if 'close' not in self.data.columns:
            raise ValueError("Colonne 'close' manquante pour le RSI.")
        self.data['RSI'] = ta.momentum.rsi(self.data['close'], window=self.rsi_period)

    def add_stochastic(self):
        """Ajoute Stochastic Oscillator."""
        if not all(col in self.data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Colonnes 'high', 'low', 'close' manquantes pour le Stochastic.")
        self.data['Stoch_K'] = ta.momentum.stoch(self.data['high'], self.data['low'], self.data['close'], window=self.stoch_k, smooth_window=self.stoch_d)
        self.data['Stoch_D'] = ta.momentum.stoch_signal(self.data['high'], self.data['low'], self.data['close'], window=self.stoch_k, smooth_window=self.stoch_d)

    def add_williams_r(self):
        """Ajoute Williams %R."""
        if not all(col in self.data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Colonnes 'high', 'low', 'close' manquantes pour le Williams %R.")
        self.data['WilliamsR'] = ta.momentum.williams_r(self.data['high'], self.data['low'], self.data['close'], lbp=self.williams_window)

    def add_bollinger_bands(self):
        """Ajoute les bandes de Bollinger."""
        if 'close' not in self.data.columns:
            raise ValueError("Colonne 'close' manquante pour les bandes de Bollinger.")
        self.data['BB_Upper'] = ta.volatility.bollinger_hband(self.data['close'], window=self.bb_window, window_dev=self.bb_std)
        self.data['BB_Middle'] = ta.volatility.bollinger_mavg(self.data['close'], window=self.bb_window)
        self.data['BB_Lower'] = ta.volatility.bollinger_lband(self.data['close'], window=self.bb_window, window_dev=self.bb_std)

    def add_atr(self):
        """Ajoute l'Average True Range (ATR)."""
        if not all(col in self.data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Colonnes 'high', 'low', 'close' manquantes pour l'ATR.")
        self.data['ATR'] = ta.volatility.average_true_range(
            self.data['high'], self.data['low'], self.data['close'], window=self.atr_window
        )

    def add_adx(self):
        """Ajoute l'ADX (Average Directional Index)."""
        if not all(col in self.data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Colonnes 'high', 'low', 'close' manquantes pour l'ADX.")
        self.data['ADX'] = ta.trend.adx(self.data['high'], self.data['low'], self.data['close'], window=self.adx_window)

    def add_volume_indicators(self):
        """Ajoute les indicateurs de volume."""
        if not all(col in self.data.columns for col in ['close', 'volume', 'high', 'low']):
            raise ValueError("Colonnes nécessaires manquantes pour les indicateurs de volume.")
        self.data['OBV'] = ta.volume.on_balance_volume(self.data['close'], self.data['volume'])
        self.data['VWAP'] = ta.volume.volume_weighted_average_price(
            self.data['high'], self.data['low'], self.data['close'], self.data['volume']
        )

    def add_support_resistance(self):
        """Calcule les niveaux de support et résistance (expérimental, binaire)."""
        if 'close' not in self.data.columns:
            return
        window = self.support_resistance_window
        supports = []
        resistances = []
        closes = self.data['close'].values
        for i in range(len(closes)):
            if i < window or i > len(closes) - window - 1:
                supports.append(0)
                resistances.append(0)
            else:
                is_support = all(closes[i] <= closes[i-j] for j in range(1, window+1)) and \
                             all(closes[i] <= closes[i+j] for j in range(1, window+1))
                is_resistance = all(closes[i] >= closes[i-j] for j in range(1, window+1)) and \
                                all(closes[i] >= closes[i+j] for j in range(1, window+1))
                supports.append(1 if is_support else 0)
                resistances.append(1 if is_resistance else 0)
        self.data['Support'] = supports
        self.data['Resistance'] = resistances
=======
from typing import Dict, Tuple, List

class EnhancedTechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def add_all_indicators(self) -> pd.DataFrame:
        """Ajoute tous les indicateurs techniques"""
        # Tendance
        self.add_moving_averages()
        self.add_macd()
        self.add_adx()
        
        # Momentum
        self.add_rsi()
        self.add_stochastic()
        self.add_williams_r()
        
        # Volatilité
        self.add_bollinger_bands()
        self.add_atr()
        
        # Volume
        self.add_volume_indicators()
        
        return self.data
    
    def add_moving_averages(self):
        """Ajoute plusieurs moyennes mobiles"""
        for period in [20, 50, 100, 200]:
            self.data[f'SMA_{period}'] = ta.trend.sma_indicator(self.data['close'], period)
            self.data[f'EMA_{period}'] = ta.trend.ema_indicator(self.data['close'], period)
    
    def add_macd(self):
        """Ajoute MACD avec histogramme"""
        self.data['MACD'] = ta.trend.macd(self.data['close'])
        self.data['MACD_Signal'] = ta.trend.macd_signal(self.data['close'])
        self.data['MACD_Hist'] = ta.trend.macd_diff(self.data['close'])
    
    def add_rsi(self):
        """Ajoute RSI et Stochastic RSI"""
        self.data['RSI'] = ta.momentum.rsi(self.data['close'])
        
    def add_bollinger_bands(self):
        """Ajoute les bandes de Bollinger"""
        self.data['BB_Upper'] = ta.volatility.bollinger_hband(self.data['close'])
        self.data['BB_Middle'] = ta.volatility.bollinger_mavg(self.data['close'])
        self.data['BB_Lower'] = ta.volatility.bollinger_lband(self.data['close'])
        
    def add_volume_indicators(self):
        """Ajoute les indicateurs de volume"""
        self.data['OBV'] = ta.volume.on_balance_volume(self.data['close'], self.data['volume'])
        self.data['Volume_SMA'] = ta.volume.volume_weighted_average_price(
            self.data['high'], 
            self.data['low'], 
            self.data['close'], 
            self.data['volume']
        )
    
    def add_support_resistance(self):
        """Calcule les niveaux de support et résistance"""
        def find_levels(prices: pd.Series, window: int = 20) -> List[float]:
            levels = []
            for i in range(window, len(prices) - window):
                if self._is_support(prices, i, window):
                    levels.append(prices[i])
                elif self._is_resistance(prices, i, window):
                    levels.append(prices[i])
            return levels
        
        self.data['Support_Resistance'] = find_levels(self.data['close'])
    
    @staticmethod
    def _is_support(prices: pd.Series, i: int, window: int) -> bool:
        """Détermine si un point est un support"""
        return (
            all(prices[i] <= prices[i-j] for j in range(1, window+1)) and
            all(prices[i] <= prices[i+j] for j in range(1, window+1))
        )
    
    @staticmethod
    def _is_resistance(prices: pd.Series, i: int, window: int) -> bool:
        """Détermine si un point est une résistance"""
        return (
            all(prices[i] >= prices[i-j] for j in range(1, window+1)) and
            all(prices[i] >= prices[i+j] for j in range(1, window+1))
        ) 
>>>>>>> 2872012 (Initial commit: Bitcoin Analysis Dashboard)
