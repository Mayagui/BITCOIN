import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
import time
import os

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

class BitcoinDataCollector:
    def __init__(
        self,
        db_path: str = 'bitcoin_data.db',
        symbol: str = "BTC-USD",
        interval: str = "1h",
        realtime_sleep: int = 60,
        use_mongo: bool = False,
        mongo_uri: str = None,
        mongo_db: str = "bitcoin_db"
    ):
        """
        Initialise le collecteur de données Bitcoin.
        """
        self.db_path = db_path
        self.symbol = symbol
        self.interval = interval
        self.realtime_sleep = realtime_sleep
        self.use_mongo = use_mongo and MONGO_AVAILABLE and mongo_uri is not None
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self._setup_logging()
        if self.use_mongo:
            self._init_mongo()
        else:
            self.init_database()

    def _setup_logging(self):
        """
        Configure le logging avancé (console + fichier).
        """
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{log_dir}/data_collector.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _init_mongo(self):
        """
        Initialise la connexion MongoDB.
        """
        try:
            self.mongo_client = MongoClient(self.mongo_uri)
            self.db = self.mongo_client[self.mongo_db]
            self.historical_collection = self.db["historical_data"]
            self.realtime_collection = self.db["realtime_data"]
            self.logger.info("Connexion MongoDB réussie.")
        except Exception as e:
            self.logger.error(f"Erreur de connexion MongoDB: {e}")
            self.use_mongo = False
            self.init_database()

    def init_database(self):
        """
        Initialise la base de données SQLite.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
                    volume REAL
                )
            ''')
            conn.commit()

    def fetch_historical_data(self, years: int = 5) -> pd.DataFrame:
        """
        Récupère les données historiques via yfinance et les stocke.
        """
        try:
            end = datetime.now()
            start = end - timedelta(days=years*365)
            self.logger.info(f"Récupération de l'historique {self.symbol} via yfinance...")
            df = yf.download(self.symbol, start=start, end=end, interval=self.interval)
            if df.empty:
                self.logger.warning("Aucune donnée récupérée via yfinance.")
                return pd.DataFrame()
            df = df.reset_index()
            df.rename(columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            self.store_historical_data(df)
            return df
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données historiques: {e}")
            return pd.DataFrame()

    def store_historical_data(self, data: pd.DataFrame):
        """
        Stocke les données historiques dans SQLite ou MongoDB.
        """
        try:
            if self.use_mongo:
                records = data.to_dict(orient="records")
                self.historical_collection.delete_many({})
                if records:
                    self.historical_collection.insert_many(records)
                self.logger.info(f"Données historiques stockées dans MongoDB: {len(records)} entrées")
            else:
                with sqlite3.connect(self.db_path) as conn:
                    data.to_sql('historical_data', conn, if_exists='replace', index=False)
                self.logger.info(f"Données historiques stockées dans SQLite: {len(data)} entrées")
        except Exception as e:
            self.logger.error(f"Erreur lors du stockage: {e}")

    def clean_old_realtime_data(self, days: int = 7):
        """
        Supprime les données temps réel de plus de X jours.
        """
        try:
            cutoff = datetime.now() - timedelta(days=days)
            if self.use_mongo:
                result = self.realtime_collection.delete_many({"timestamp": {"$lt": cutoff}})
                self.logger.info(f"{result.deleted_count} entrées temps réel supprimées de MongoDB.")
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM realtime_data WHERE timestamp < ?",
                        (cutoff.strftime("%Y-%m-%d %H:%M:%S"),)
                    )
                    conn.commit()
                self.logger.info(f"Données temps réel de plus de {days} jours supprimées (SQLite).")
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {e}")

    def realtime_data_stream(self):
        """
        Récupère le prix actuel toutes les X secondes via yfinance.
        """
        self.logger.info(f"Lancement du flux temps réel pour {self.symbol} (intervalle {self.realtime_sleep}s)")
        while True:
            try:
                ticker = yf.Ticker(self.symbol)
                price = ticker.history(period="1m", interval="1m")
                if not price.empty:
                    last = price.iloc[-1]
                    data = {
                        'timestamp': last.name.to_pydatetime(),
                        'price': last['Close'],
                        'volume': last['Volume']
                    }
                    if self.use_mongo:
                        self.realtime_collection.replace_one(
                            {"timestamp": data['timestamp']},
                            data,
                            upsert=True
                        )
                        self.logger.info(f"Donnée temps réel insérée dans MongoDB: {data}")
                    else:
                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute('''
                                INSERT OR REPLACE INTO realtime_data 
                                (timestamp, price, volume)
                                VALUES (?, ?, ?)
                            ''', (data['timestamp'], data['price'], data['volume']))
                            conn.commit()
                        self.logger.info(f"Donnée temps réel insérée dans SQLite: {data}")
                else:
                    self.logger.warning("Aucune donnée temps réel récupérée.")
                time.sleep(self.realtime_sleep)
            except Exception as e:
                self.logger.error(f"Erreur dans le flux temps réel: {e}")
                time.sleep(5)

    def get_latest_realtime(self):
        """
        Récupère la dernière donnée temps réel stockée.
        """
        try:
            if self.use_mongo:
                doc = self.realtime_collection.find_one(sort=[("timestamp", -1)])
                return pd.DataFrame([doc]) if doc else pd.DataFrame()
            else:
                with sqlite3.connect(self.db_path) as conn:
                    df = pd.read_sql_query(
                        "SELECT * FROM realtime_data ORDER BY timestamp DESC LIMIT 1", conn
                    )
                return df
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture du temps réel: {e}")
            return pd.DataFrame()
