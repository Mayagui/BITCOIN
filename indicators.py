import pandas as pd
import numpy as np
import ta
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