import pandas as pd
import ta
from utils import load_bitcoin_data_live

def add_indicators(df):
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    return df

def load_bitcoin_data_live():
    # Ta fonction ici...