import ccxt
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
import asyncio
from typing import Dict, Any

class BitcoinDataCollector:
    def __init__(self):
        # Initialisation de la base de données SQLite
        self.db_path = 'bitcoin_data.db'
        self.init_database()
        
        # Initialisation de l'exchange (Binance)
        self.exchange = ccxt.binance()
        
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def init_database(self):
        """Initialise la base de données SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Création des tables si elles n'existent pas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_data (
                timestamp DATETIME PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_data (
                timestamp DATETIME PRIMARY KEY,
                price REAL,
                volume REAL,
                high REAL,
                low REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    def fetch_historical_data(self, years: int = 5) -> pd.DataFrame:
        """Récupère les données historiques"""
        try:
            start_date = int((datetime.now() - timedelta(days=years*365)).timestamp() * 1000)
            all_data = []
            
            self.logger.info("Récupération des données historiques...")
            ohlcv = self.exchange.fetch_ohlcv(
                symbol='BTC/USDT',
                timeframe='1h',
                since=start_date,
                limit=1000
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sauvegarde dans SQLite
            self.store_historical_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return pd.DataFrame()

    def store_historical_data(self, data: pd.DataFrame):
        """Stocke les données historiques dans SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            data.to_sql('historical_data', conn, if_exists='replace', index=False)
            conn.close()
            self.logger.info(f"Données historiques stockées: {len(data)} entrées")
        except Exception as e:
            self.logger.error(f"Erreur lors du stockage: {e}")

    async def realtime_data_stream(self):
        """Génère un flux de données en temps réel"""
        while True:
            try:
                ticker = self.exchange.fetch_ticker('BTC/USDT')
                data = {
                    'timestamp': datetime.now(),
                    'price': ticker['last'],
                    'volume': ticker['quoteVolume'],
                    'high': ticker['high'],
                    'low': ticker['low']
                }
                
                # Stockage dans SQLite
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO realtime_data 
                    (timestamp, price, volume, high, low)
                    VALUES (?, ?, ?, ?, ?)
                ''', (data['timestamp'], data['price'], data['volume'], data['high'], data['low']))
                conn.commit()
                conn.close()
                
                await asyncio.sleep(1)  # Mise à jour toutes les secondes
                
            except Exception as e:
                self.logger.error(f"Erreur dans le flux temps réel: {e}")
                await asyncio.sleep(5) 