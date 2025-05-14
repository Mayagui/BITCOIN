import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
import os
import pickle
from typing import Dict, List, Optional, Tuple
import asyncio
import sqlite3
import logging
import json
from dataclasses import dataclass
import requests
import aiohttp
from textblob import TextBlob
from pytrends.request import TrendReq

# Configuration de la page
st.set_page_config(
    page_title="Bitcoin Analytics Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stMetric {
        background-color: #1E2130;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric:hover {
        transform: translateY(-5px);
        transition: all 0.3s ease;
    }
    .indicator-box {
        background-color: #1E2130;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .signal-buy {
        color: #00FF00;
        font-weight: bold;
    }
    .signal-sell {
        color: #FF0000;
        font-weight: bold;
    }
    .signal-neutral {
        color: #FFA500;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@dataclass
class IndicatorSignal:
    """Classe pour stocker les signaux des indicateurs"""
    name: str
    value: float
    signal: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0 à 1
    description: str

@dataclass
class SentimentScore:
    """Classe pour stocker les scores de sentiment"""
    source: str
    score: float  # -1 à 1
    confidence: float  # 0 à 1
    timestamp: datetime
    raw_data: Dict

class EnhancedTechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.logger = logging.getLogger(__name__)
        
    def add_all_indicators(self) -> pd.DataFrame:
        """Ajoute tous les indicateurs techniques avec gestion des erreurs"""
        try:
            # Tendance
            self.add_moving_averages()
            self.add_macd()
            self.add_adx()
            self.add_ichimoku()
            
            # Momentum
            self.add_rsi()
            self.add_stochastic()
            self.add_williams_r()
            self.add_momentum()
            self.add_cci()
            
            # Volatilité
            self.add_bollinger_bands()
            self.add_atr()
            self.add_keltner_channels()
            
            # Volume
            self.add_volume_indicators()
            
            # Support/Résistance
            self.add_support_resistance()
            
            # Patterns
            self.add_candlestick_patterns()
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des indicateurs: {e}")
            return self.data
    
    def add_moving_averages(self):
        """Ajoute plusieurs moyennes mobiles avec signaux"""
        periods = [20, 50, 100, 200]
        for period in periods:
            # SMA
            self.data[f'SMA_{period}'] = ta.trend.sma_indicator(self.data['close'], period)
            # EMA
            self.data[f'EMA_{period}'] = ta.trend.ema_indicator(self.data['close'], period)
            # WMA (Weighted Moving Average)
            self.data[f'WMA_{period}'] = ta.trend.wma_indicator(self.data['close'], period)
            
            # Signaux de croisement
            if period > 20:  # Comparer avec SMA 20
                self.data[f'SMA_Cross_{period}'] = np.where(
                    self.data[f'SMA_{period}'] > self.data['SMA_20'],
                    'bullish',
                    'bearish'
                )
    
    def add_macd(self):
        """Ajoute MACD avec signaux améliorés"""
        # MACD standard
        self.data['MACD'] = ta.trend.macd(self.data['close'])
        self.data['MACD_Signal'] = ta.trend.macd_signal(self.data['close'])
        self.data['MACD_Hist'] = ta.trend.macd_diff(self.data['close'])
        
        # MACD avec paramètres personnalisés
        self.data['MACD_Fast'] = ta.trend.macd(self.data['close'], window_slow=26, window_fast=12)
        self.data['MACD_Slow'] = ta.trend.macd(self.data['close'], window_slow=52, window_fast=26)
        
        # Signaux MACD
        self.data['MACD_Signal_Type'] = np.where(
            self.data['MACD'] > self.data['MACD_Signal'],
            'bullish',
            'bearish'
        )
    
    def add_rsi(self):
        """Ajoute RSI et Stochastic RSI avec signaux"""
        # RSI standard
        self.data['RSI'] = ta.momentum.rsi(self.data['close'])
        
        # Stochastic RSI
        self.data['Stoch_RSI'] = ta.momentum.stochrsi(self.data['close'])
        
        # RSI avec plusieurs périodes
        for period in [14, 21, 50]:
            self.data[f'RSI_{period}'] = ta.momentum.rsi(self.data['close'], window=period)
        
        # Signaux RSI
        self.data['RSI_Signal'] = np.where(
            self.data['RSI'] < 30,
            'oversold',
            np.where(self.data['RSI'] > 70, 'overbought', 'neutral')
        )
    
    def add_bollinger_bands(self):
        """Ajoute les bandes de Bollinger avec signaux"""
        # Bandes standard
        self.data['BB_Upper'] = ta.volatility.bollinger_hband(self.data['close'])
        self.data['BB_Middle'] = ta.volatility.bollinger_mavg(self.data['close'])
        self.data['BB_Lower'] = ta.volatility.bollinger_lband(self.data['close'])
        
        # Bandes avec différents écarts-types
        for std in [1, 2, 3]:
            self.data[f'BB_Upper_{std}'] = ta.volatility.bollinger_hband(self.data['close'], window=20, window_dev=std)
            self.data[f'BB_Lower_{std}'] = ta.volatility.bollinger_lband(self.data['close'], window=20, window_dev=std)
        
        # Position relative au prix
        self.data['BB_Position'] = (self.data['close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        
        # Signaux
        self.data['BB_Signal'] = np.where(
            self.data['BB_Position'] > 0.8,
            'overbought',
            np.where(self.data['BB_Position'] < 0.2, 'oversold', 'neutral')
        )
    
    def add_volume_indicators(self):
        """Ajoute les indicateurs de volume avec signaux"""
        # OBV (On Balance Volume)
        self.data['OBV'] = ta.volume.on_balance_volume(self.data['close'], self.data['volume'])
        
        # VWAP (Volume Weighted Average Price)
        self.data['VWAP'] = ta.volume.volume_weighted_average_price(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            self.data['volume']
        )
        
        # Chaikin Money Flow
        self.data['CMF'] = ta.volume.chaikin_money_flow(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            self.data['volume']
        )
        
        # Volume SMA
        self.data['Volume_SMA'] = ta.volume.volume_weighted_average_price(
            self.data['high'],
            self.data['low'],
            self.data['close'],
            self.data['volume']
        )
        
        # Signaux de volume
        self.data['Volume_Signal'] = np.where(
            self.data['volume'] > self.data['Volume_SMA'] * 1.5,
            'high_volume',
            'normal_volume'
        )
    
    def add_ichimoku(self):
        """Ajoute l'indicateur Ichimoku Cloud"""
        ichimoku = ta.trend.IchimokuIndicator(
            high=self.data['high'],
            low=self.data['low']
        )
        
        self.data['Ichimoku_A'] = ichimoku.ichimoku_a()
        self.data['Ichimoku_B'] = ichimoku.ichimoku_b()
        self.data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        self.data['Ichimoku_Conv'] = ichimoku.ichimoku_conversion_line()
        self.data['Ichimoku_Span'] = ichimoku.ichimoku_span()
    
    def add_candlestick_patterns(self):
        """Ajoute la détection des patterns de chandeliers"""
        # Doji
        self.data['Doji'] = np.where(
            abs(self.data['open'] - self.data['close']) <= (self.data['high'] - self.data['low']) * 0.1,
            True,
            False
        )
        
        # Hammer
        self.data['Hammer'] = np.where(
            (self.data['close'] > self.data['open']) &
            ((self.data['high'] - self.data['low']) > 3 * (self.data['open'] - self.data['low'])) &
            ((self.data['close'] - self.data['low']) / (0.001 + self.data['high'] - self.data['low']) > 0.6) &
            ((self.data['open'] - self.data['low']) / (0.001 + self.data['high'] - self.data['low']) > 0.6),
            True,
            False
        )
    
    def get_signals(self) -> List[IndicatorSignal]:
        """Récupère tous les signaux des indicateurs"""
        signals = []
        
        # RSI
        rsi = self.data['RSI'].iloc[-1]
        if rsi < 30:
            signals.append(IndicatorSignal(
                name='RSI',
                value=rsi,
                signal='buy',
                strength=1 - (rsi / 30),
                description='RSI en zone de survente'
            ))
        elif rsi > 70:
            signals.append(IndicatorSignal(
                name='RSI',
                value=rsi,
                signal='sell',
                strength=(rsi - 70) / 30,
                description='RSI en zone de surachat'
            ))
        
        # MACD
        if self.data['MACD'].iloc[-1] > self.data['MACD_Signal'].iloc[-1]:
            signals.append(IndicatorSignal(
                name='MACD',
                value=self.data['MACD'].iloc[-1],
                signal='buy',
                strength=0.7,
                description='Signal d\'achat MACD'
            ))
        
        # Bollinger Bands
        bb_pos = self.data['BB_Position'].iloc[-1]
        if bb_pos < 0.2:
            signals.append(IndicatorSignal(
                name='Bollinger Bands',
                value=bb_pos,
                signal='buy',
                strength=1 - (bb_pos / 0.2),
                description='Prix sous la bande inférieure'
            ))
        
        return signals
    
    def get_signal_summary(self) -> Dict:
        """Récupère un résumé des signaux"""
        signals = self.get_signals()
        
        return {
            'buy_signals': len([s for s in signals if s.signal == 'buy']),
            'sell_signals': len([s for s in signals if s.signal == 'sell']),
            'strongest_buy': max([s for s in signals if s.signal == 'buy'], key=lambda x: x.strength, default=None),
            'strongest_sell': max([s for s in signals if s.signal == 'sell'], key=lambda x: x.strength, default=None),
            'all_signals': signals
        }

class BitcoinDataCollector:
    def __init__(self):
        """Initialise le collecteur de données avec plusieurs exchanges"""
        self.exchanges = {
            'binance': ccxt.binance(),
            'coinbase': ccxt.coinbasepro(),
            'kraken': ccxt.kraken()
        }
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Configuration des timeframes disponibles
        self.timeframes = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }

    def get_cached_data(self, key: str, max_age: int = 3600) -> Optional[pd.DataFrame]:
        """
        Récupère les données du cache si elles sont récentes
        
        Args:
            key: Clé unique pour les données
            max_age: Âge maximum du cache en secondes
        
        Returns:
            DataFrame si les données sont en cache et récentes, None sinon
        """
        cache_file = f"{self.cache_dir}/{key}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            if time.time() - cache['timestamp'] < max_age:
                return cache['data']
        return None

    def store_in_cache(self, key: str, data: pd.DataFrame):
        """
        Stocke les données dans le cache
        
        Args:
            key: Clé unique pour les données
            data: DataFrame à stocker
        """
        cache_file = f"{self.cache_dir}/{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'timestamp': time.time(),
                'data': data
            }, f)

    def fetch_historical_data(self, years: int = 5, timeframe: str = '1h') -> pd.DataFrame:
        """
        Récupère les données historiques avec gestion du cache et des limites API
        
        Args:
            years: Nombre d'années de données à récupérer
            timeframe: Intervalle de temps ('1m', '5m', '15m', '1h', '4h', '1d')
        
        Returns:
            DataFrame contenant les données OHLCV
        """
        try:
            # Vérifier le cache
            cache_key = f"historical_data_{timeframe}_{years}y"
            cached_data = self.get_cached_data(cache_key)
            if cached_data is not None:
                self.logger.info("Utilisation des données en cache")
                return cached_data

            start_date = int((datetime.now() - timedelta(days=years*365)).timestamp() * 1000)
            all_data = []
            
            self.logger.info(f"Récupération des données historiques ({timeframe})...")
            
            # Récupération par lots de 1000 points
            while start_date < int(datetime.now().timestamp() * 1000):
                ohlcv = self.exchanges['binance'].fetch_ohlcv(
                    symbol='BTC/USDT',
                    timeframe=timeframe,
                    since=start_date,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                start_date = ohlcv[-1][0] + 1
                
                # Respecter les limites de l'API
                time.sleep(1)
            
            if not all_data:
                self.logger.warning("Aucune donnée récupérée")
                return pd.DataFrame()
            
            # Création du DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ajout de métriques calculées
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Sauvegarder dans le cache et la base de données
            self.store_in_cache(cache_key, df)
            self.store_historical_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame()
    
    def get_market_depth(self, exchange: str = 'binance') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Récupère la profondeur du marché (order book)
        
        Returns:
            Tuple de DataFrames (bids, asks)
        """
        try:
            exchange = self.exchanges[exchange]
            order_book = exchange.fetch_order_book('BTC/USDT')
            
            bids = pd.DataFrame(order_book['bids'], columns=['price', 'amount'])
            asks = pd.DataFrame(order_book['asks'], columns=['price', 'amount'])
            
            return bids, asks
            
        except Exception as e:
            st.error(f"Erreur lors de la récupération de la profondeur du marché: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_market_metrics(self, exchange: str = 'binance') -> Dict:
        """
        Récupère les métriques de marché actuelles
        
        Returns:
            Dictionnaire contenant les métriques
        """
        try:
            exchange = self.exchanges[exchange]
            ticker = exchange.fetch_ticker('BTC/USDT')
            
            return {
                'price': ticker['last'],
                'change_24h': ticker['percentage'],
                'volume_24h': ticker['quoteVolume'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'spread': ticker['ask'] - ticker['bid'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Erreur lors de la récupération des métriques de marché: {str(e)}")
            return {}

    async def realtime_data_stream(self, callback: Optional[callable] = None):
        """
        Génère un flux de données en temps réel avec gestion des erreurs améliorée
        
        Args:
            callback: Fonction à appeler avec les nouvelles données
        """
        consecutive_errors = 0
        max_errors = 5
        
        while True:
            try:
                ticker = self.exchanges['binance'].fetch_ticker('BTC/USDT')
                data = {
                    'timestamp': datetime.now(),
                    'price': ticker['last'],
                    'volume': ticker['quoteVolume'],
                    'high': ticker['high'],
                    'low': ticker['low'],
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'spread': ticker['ask'] - ticker['bid']
                }
                
                # Stockage dans SQLite
                self.store_realtime_data(data)
                
                # Appel du callback si fourni
                if callback:
                    await callback(data)
                
                consecutive_errors = 0
                await asyncio.sleep(1)
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Erreur dans le flux temps réel: {e}")
                
                if consecutive_errors >= max_errors:
                    self.logger.error("Trop d'erreurs consécutives, pause de 5 minutes")
                    await asyncio.sleep(300)
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(5)

def display_market_data(df: pd.DataFrame, metrics: Dict):
    """Affiche les données de marché avec Plotly"""
    # Créer un graphique avec deux sous-graphiques
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Prix Bitcoin', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Ajouter le graphique en chandeliers
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC/USDT'
        ),
        row=1, col=1
    )
    
    # Ajouter le volume
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume'
        ),
        row=2, col=1
    )
    
    # Mettre à jour la mise en page
    fig.update_layout(
        title='Analyse Bitcoin en Temps Réel',
        yaxis_title='Prix (USDT)',
        yaxis2_title='Volume',
        template='plotly_dark',
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def main():
    st.title("📊 Analyse du Marché Bitcoin")
    
    # Initialisation du collecteur de données
    collector = BitcoinDataCollector()
    
    # Sidebar pour les paramètres
    st.sidebar.header("Paramètres")
    timeframe = st.sidebar.selectbox(
        "Intervalle de temps",
        ['1m', '5m', '15m', '1h', '4h', '1d'],
        index=3
    )
    exchange = st.sidebar.selectbox(
        "Exchange",
        ['binance', 'coinbase', 'kraken'],
        index=0
    )
    
    # Récupération des données
    df = collector.fetch_historical_data(years=5, timeframe=timeframe)
    metrics = collector.get_market_metrics(exchange=exchange)
    
    if not df.empty and metrics:
        # Affichage des métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Prix Bitcoin",
                f"${metrics['price']:,.2f}",
                f"{metrics['change_24h']:+.2f}%"
            )
        with col2:
            st.metric(
                "Volume 24h",
                f"${metrics['volume_24h']:,.0f}"
            )
        with col3:
            st.metric(
                "Spread",
                f"${metrics['spread']:,.2f}"
            )
        with col4:
            st.metric(
                "Amplitude 24h",
                f"${metrics['high_24h'] - metrics['low_24h']:,.2f}"
            )
        
        # Affichage du graphique
        fig = display_market_data(df, metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage des statistiques
        st.subheader("Statistiques")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Statistiques de prix")
            st.dataframe(df['close'].describe())
        with col2:
            st.write("Statistiques de volume")
            st.dataframe(df['volume'].describe())
        
        # Dernière mise à jour
        st.caption(f"Dernière mise à jour: {metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

class DatabaseManager:
    def __init__(self, db_path='bitcoin_data.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """Initialise la base de données avec des tables optimisées"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Table pour les données de marché (avec index)
        c.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp DATETIME PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                returns REAL,
                volatility REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Index pour optimiser les requêtes
        c.execute('CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)')
        
        # Table pour les sentiments (avec index)
        c.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                timestamp DATETIME PRIMARY KEY,
                fear_greed_value REAL,
                fear_greed_classification TEXT,
                market_sentiment REAL,
                google_trends_value REAL,
                social_media_sentiment REAL,
                news_sentiment REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table pour Google Trends (avec index)
        c.execute('''
            CREATE TABLE IF NOT EXISTS google_trends (
                timestamp DATETIME PRIMARY KEY,
                bitcoin REAL,
                crypto REAL,
                BTC REAL,
                ethereum REAL,
                blockchain REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table pour les indicateurs techniques
        c.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                timestamp DATETIME PRIMARY KEY,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                stoch_rsi REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (timestamp) REFERENCES market_data(timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_latest_data(self, limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Récupère les dernières données avec jointures optimisées
        
        Args:
            limit: Nombre de lignes à récupérer
            
        Returns:
            Dictionnaire contenant les DataFrames des différentes tables
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Requête optimisée avec jointure
            query = '''
                SELECT 
                    m.*,
                    t.rsi, t.macd, t.macd_signal, t.macd_hist,
                    t.bb_upper, t.bb_middle, t.bb_lower, t.stoch_rsi,
                    s.fear_greed_value, s.market_sentiment,
                    g.bitcoin as google_trends_bitcoin
                FROM market_data m
                LEFT JOIN technical_indicators t ON m.timestamp = t.timestamp
                LEFT JOIN sentiment_data s ON m.timestamp = s.timestamp
                LEFT JOIN google_trends g ON m.timestamp = g.timestamp
                ORDER BY m.timestamp DESC
                LIMIT ?
            '''
            
            df = pd.read_sql(query, conn, params=(limit,))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            conn.close()
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame()
    
    def save_market_data(self, data: pd.DataFrame) -> bool:
        """
        Stocke les données de marché avec gestion des doublons
        
        Args:
            data: DataFrame contenant les données de marché
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Vérifier les doublons
            existing_timestamps = pd.read_sql(
                'SELECT timestamp FROM market_data',
                conn
            )['timestamp'].tolist()
            
            # Filtrer les nouvelles données
            new_data = data[~data['timestamp'].isin(existing_timestamps)]
            
            if not new_data.empty:
                new_data.to_sql(
                    'market_data',
                    conn,
                    if_exists='append',
                    index=False
                )
                self.logger.info(f"{len(new_data)} nouvelles entrées ajoutées")
            
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des données: {e}")
            return False
    
    def get_data_range(self, 
                      start_date: datetime,
                      end_date: datetime,
                      timeframe: str = '1h') -> pd.DataFrame:
        """
        Récupère les données sur une période spécifique
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            timeframe: Intervalle de temps ('1h', '4h', '1d')
            
        Returns:
            DataFrame contenant les données de la période
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    m.*,
                    t.rsi, t.macd, t.macd_signal, t.macd_hist,
                    t.bb_upper, t.bb_middle, t.bb_lower, t.stoch_rsi,
                    s.fear_greed_value, s.market_sentiment,
                    g.bitcoin as google_trends_bitcoin
                FROM market_data m
                LEFT JOIN technical_indicators t ON m.timestamp = t.timestamp
                LEFT JOIN sentiment_data s ON m.timestamp = s.timestamp
                LEFT JOIN google_trends g ON m.timestamp = g.timestamp
                WHERE m.timestamp BETWEEN ? AND ?
                ORDER BY m.timestamp
            '''
            
            df = pd.read_sql(
                query,
                conn,
                params=(start_date, end_date)
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            conn.close()
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame()
    
    def save_technical_indicators(self, data: pd.DataFrame) -> bool:
        """
        Stocke les indicateurs techniques
        
        Args:
            data: DataFrame contenant les indicateurs techniques
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            data.to_sql(
                'technical_indicators',
                conn,
                if_exists='append',
                index=False
            )
            
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des indicateurs: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """
        Nettoie les anciennes données
        
        Args:
            days_to_keep: Nombre de jours de données à conserver
            
        Returns:
            bool: True si le nettoyage a réussi
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Supprimer les anciennes données de toutes les tables
            tables = ['market_data', 'sentiment_data', 'google_trends', 'technical_indicators']
            for table in tables:
                cursor.execute(f'''
                    DELETE FROM {table}
                    WHERE timestamp < ?
                ''', (cutoff_date,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Données plus anciennes que {days_to_keep} jours supprimées")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage des données: {e}")
            return False

class MarketSentimentAnalyzer:
    def __init__(self):
        # URLs des APIs
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.cache_dir = "sentiment_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Configuration du logging
        self.logger = logging.getLogger(__name__)
        
        # Initialisation de Google Trends
        self.init_google_trends()
        
        # Configuration des poids pour l'agrégation
        self.sentiment_weights = {
            'fear_greed': 0.4,
            'news': 0.3,
            'market_metrics': 0.2,
            'google_trends': 0.1
        }

    def init_google_trends(self):
        """Initialise le client Google Trends"""
        try:
            self.pytrends = TrendReq(hl='fr-FR', tz=360)
        except Exception as e:
            self.logger.error(f"Erreur d'initialisation Google Trends: {e}")
            self.pytrends = None

    async def get_fear_greed_index(self) -> SentimentScore:
        """Obtient l'indice Fear & Greed avec cache"""
        try:
            # Vérifier le cache
            cache_file = f"{self.cache_dir}/fear_greed.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                if datetime.now().timestamp() - cache['timestamp'] < 3600:  # Cache de 1h
                    return SentimentScore(**cache['data'])

            async with aiohttp.ClientSession() as session:
                async with session.get(self.fear_greed_url) as response:
                    data = await response.json()
                    
                    score = float(data['data'][0]['value'])
                    normalized_score = (score - 50) / 50  # Normalise à -1 à 1
                    
                    sentiment = SentimentScore(
                        source='fear_greed',
                        score=normalized_score,
                        confidence=0.8,
                        timestamp=datetime.now(),
                        raw_data=data
                    )
                    
                    # Sauvegarder dans le cache
                    with open(cache_file, 'w') as f:
                        json.dump({
                            'timestamp': datetime.now().timestamp(),
                            'data': sentiment.__dict__
                        }, f)
                    
                    return sentiment
                    
        except Exception as e:
            self.logger.error(f"Erreur Fear & Greed Index: {e}")
            return SentimentScore(
                source='fear_greed',
                score=0,
                confidence=0,
                timestamp=datetime.now(),
                raw_data={}
            )

    async def get_news_sentiment(self) -> SentimentScore:
        """Analyse le sentiment des actualités via CoinGecko"""
        try:
            async with aiohttp.ClientSession() as session:
                # Utilisation de l'API CoinGecko qui est gratuite et sans clé
                url = f"{self.coingecko_url}/coins/bitcoin/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': '1',
                    'interval': 'hourly'
                }
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    # Analyse des variations de prix
                    prices = [price[1] for price in data['prices']]
                    volumes = [volume[1] for volume in data['total_volumes']]
                    
                    # Calcul des variations
                    price_changes = np.diff(prices) / prices[:-1]
                    volume_changes = np.diff(volumes) / volumes[:-1]
                    
                    # Score basé sur les variations
                    price_score = np.mean(price_changes)
                    volume_score = np.mean(volume_changes)
                    
                    # Score final
                    final_score = (price_score * 0.7 + volume_score * 0.3)
                    
                    return SentimentScore(
                        source='market_news',
                        score=final_score,
                        confidence=0.6,
                        timestamp=datetime.now(),
                        raw_data={
                            'price_changes': price_changes.tolist(),
                            'volume_changes': volume_changes.tolist()
                        }
                    )
                    
        except Exception as e:
            self.logger.error(f"Erreur analyse actualités: {e}")
            return SentimentScore(
                source='market_news',
                score=0,
                confidence=0,
                timestamp=datetime.now(),
                raw_data={}
            )

    async def get_google_trends_sentiment(self) -> SentimentScore:
        """Analyse les tendances Google"""
        try:
            if not self.pytrends:
                raise Exception("Client Google Trends non initialisé")

            # Recherche des tendances Bitcoin
            self.pytrends.build_payload(
                ['bitcoin', 'crypto', 'BTC'],
                cat=0,
                timeframe='now 7-d',
                geo=''
            )
            
            data = self.pytrends.interest_over_time()
            if data.empty:
                return SentimentScore(
                    source='google_trends',
                    score=0,
                    confidence=0,
                    timestamp=datetime.now(),
                    raw_data={}
                )
            
            # Calcul du score basé sur la tendance
            trend_score = data['bitcoin'].mean() / 100  # Normalise à 0-1
            normalized_score = (trend_score - 0.5) * 2  # Normalise à -1 à 1
            
            return SentimentScore(
                source='google_trends',
                score=normalized_score,
                confidence=0.5,
                timestamp=datetime.now(),
                raw_data={'trend_data': data.to_dict()}
            )
            
        except Exception as e:
            self.logger.error(f"Erreur Google Trends: {e}")
            return SentimentScore(
                source='google_trends',
                score=0,
                confidence=0,
                timestamp=datetime.now(),
                raw_data={}
            )

    async def get_market_metrics(self) -> SentimentScore:
        """Obtient et analyse les métriques de marché"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.coingecko_url}/coins/bitcoin",
                    params={
                        'localization': 'false',
                        'tickers': 'false',
                        'community_data': 'true',
                        'developer_data': 'false'
                    }
                ) as response:
                    data = await response.json()
                    
                    # Calcul du score basé sur les métriques
                    price_change = data['market_data']['price_change_percentage_24h'] / 100
                    market_cap_change = data['market_data']['market_cap_change_percentage_24h'] / 100
                    
                    # Normalisation et pondération
                    score = (price_change * 0.6 + market_cap_change * 0.4)
                    
                    return SentimentScore(
                        source='market_metrics',
                        score=score,
                        confidence=0.9,
                        timestamp=datetime.now(),
                        raw_data=data
                    )
                    
        except Exception as e:
            self.logger.error(f"Erreur métriques de marché: {e}")
            return SentimentScore(
                source='market_metrics',
                score=0,
                confidence=0,
                timestamp=datetime.now(),
                raw_data={}
            )

    async def aggregate_sentiments(self) -> Dict[str, Any]:
        """Agrège tous les scores de sentiment"""
        try:
            # Récupération de tous les scores
            fear_greed = await self.get_fear_greed_index()
            news = await self.get_news_sentiment()
            market_metrics = await self.get_market_metrics()
            google_trends = await self.get_google_trends_sentiment()
            
            # Calcul du score composite pondéré
            scores = {
                'fear_greed': fear_greed,
                'news': news,
                'market_metrics': market_metrics,
                'google_trends': google_trends
            }
            
            weighted_score = sum(
                score.score * self.sentiment_weights[source] * score.confidence
                for source, score in scores.items()
            )
            
            # Normalisation finale
            final_score = max(min(weighted_score, 1), -1)
            
            return {
                'timestamp': datetime.now(),
                'composite_score': final_score,
                'individual_scores': scores,
                'sentiment': 'bullish' if final_score > 0.2 else 'bearish' if final_score < -0.2 else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur agrégation des sentiments: {e}")
            return {
                'timestamp': datetime.now(),
                'composite_score': 0,
                'individual_scores': {},
                'sentiment': 'neutral'
            }

    def get_sentiment_history(self, days: int = 30) -> pd.DataFrame:
        """Récupère l'historique des sentiments"""
        try:
            cache_file = f"{self.cache_dir}/sentiment_history.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    history = json.load(f)
                return pd.DataFrame(history)
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Erreur récupération historique: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    main()