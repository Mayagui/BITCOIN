import sqlite3
import pandas as pd
import streamlit as st
<<<<<<< HEAD
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
=======
>>>>>>> 2872012 (Initial commit: Bitcoin Analysis Dashboard)

class DatabaseManager:
    def __init__(self, db_path='bitcoin_data.db'):
        self.db_path = db_path
        self.init_database()
<<<<<<< HEAD
        self._setup_logging()

    def _setup_logging(self):
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{log_dir}/database_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def init_database(self):
        """Initialise la base de données"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Table pour les données de marché
            c.execute('''CREATE TABLE IF NOT EXISTS market_data
                        (timestamp TEXT PRIMARY KEY, open REAL, high REAL, low REAL, 
                         close REAL, volume REAL)''')
            # Table pour les sentiments
            c.execute('''CREATE TABLE IF NOT EXISTS sentiment_data
                        (timestamp TEXT PRIMARY KEY, fear_greed_value REAL, 
                         fear_greed_classification TEXT, market_sentiment REAL,
                         google_trends_value REAL)''')
            # Table pour Google Trends
            c.execute('''CREATE TABLE IF NOT EXISTS google_trends
                        (timestamp TEXT PRIMARY KEY, bitcoin REAL, crypto REAL, BTC REAL)''')
            conn.commit()

    def get_latest_data(self):
        """Récupère les dernières données"""
        with sqlite3.connect(self.db_path) as conn:
            market_data = pd.read_sql('SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 1000', conn)
            sentiment_data = pd.read_sql('SELECT * FROM sentiment_data ORDER BY timestamp DESC LIMIT 1', conn)
            trends_data = pd.read_sql('SELECT * FROM google_trends ORDER BY timestamp DESC LIMIT 24', conn)
        return market_data, sentiment_data, trends_data

    def save_market_data(self, data: pd.DataFrame):
        """Sauvegarde les données de marché (append, évite les doublons)"""
        with sqlite3.connect(self.db_path) as conn:
            data.drop_duplicates(subset=['timestamp'], inplace=True)
            data.to_sql('market_data', conn, if_exists='append', index=False)
        self.logger.info(f"{len(data)} lignes de market_data sauvegardées (append).")

    def save_sentiment_data(self, data: dict):
        """Sauvegarde les données de sentiment (append, évite les doublons)"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.DataFrame([data])
            df.drop_duplicates(subset=['timestamp'], inplace=True)
            df.to_sql('sentiment_data', conn, if_exists='append', index=False)
        self.logger.info("Donnée de sentiment sauvegardée (append).")

    def save_trends_data(self, data: pd.DataFrame):
        """Sauvegarde les données Google Trends (append, évite les doublons)"""
        with sqlite3.connect(self.db_path) as conn:
            data.drop_duplicates(subset=['timestamp'], inplace=True)
            data.to_sql('google_trends', conn, if_exists='append', index=False)
        self.logger.info("Données Google Trends sauvegardées (append).")

def train_ml_model(df: pd.DataFrame):
    """
    Entraîne un modèle ML simple sur les données de marché avec scaling.
    """
    features = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    target = (df['close'].shift(-1) > df['close']).astype(int).dropna()
    features = features.iloc[:-1]
    target = target.iloc[:len(features)]

    if features.empty or target.empty or len(features) < 10:
        raise ValueError("Pas assez de données pour entraîner le modèle ML.")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_scaled, target)
    return model, scaler

def predict_ml_signal(model, scaler, last_row: pd.DataFrame):
    """
    Prédit la tendance avec le modèle ML.
    """
    features = last_row[['open', 'high', 'low', 'close', 'volume']]
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1]
    return pred, proba

def show_ml_prediction(df: pd.DataFrame):
    if df is not None and not df.empty and len(df) > 30:
        try:
            model, scaler = train_ml_model(df)
            last_row = df.iloc[[-1]]
            ml_pred, ml_proba = predict_ml_signal(model, scaler, last_row)
            ml_label = "Hausse probable" if ml_pred == 1 else "Baisse probable"
            st.metric("Prédiction ML", ml_label, f"Confiance : {ml_proba:.2%}")
=======
        self.df = None  # Initialize df
    
    def init_database(self):
        """Initialise la base de données"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Table pour les données de marché
        c.execute('''CREATE TABLE IF NOT EXISTS market_data
                    (timestamp TEXT, open REAL, high REAL, low REAL, 
                     close REAL, volume REAL)''')
        
        # Table pour les sentiments
        c.execute('''CREATE TABLE IF NOT EXISTS sentiment_data
                    (timestamp TEXT, fear_greed_value REAL, 
                     fear_greed_classification TEXT, market_sentiment REAL,
                     google_trends_value REAL)''')
        
        # Table pour Google Trends
        c.execute('''CREATE TABLE IF NOT EXISTS google_trends
                    (timestamp TEXT, bitcoin REAL, crypto REAL, BTC REAL)''')
        
        conn.commit()
        conn.close()
    
    def get_latest_data(self):
        """Récupère les dernières données"""
        conn = sqlite3.connect(self.db_path)
        
        market_data = pd.read_sql('SELECT * FROM market_data ORDER BY timestamp DESC LIMIT 1000', conn)
        sentiment_data = pd.read_sql('SELECT * FROM sentiment_data ORDER BY timestamp DESC LIMIT 1', conn)
        trends_data = pd.read_sql('SELECT * FROM google_trends ORDER BY timestamp DESC LIMIT 24', conn)
        
        conn.close()
        return market_data, sentiment_data, trends_data

    def save_market_data(self, data):
        """Sauvegarde les données de marché"""
        conn = sqlite3.connect(self.db_path)
        data.to_sql('market_data', conn, if_exists='replace', index=False)
        conn.close()

    def save_sentiment_data(self, data):
        """Sauvegarde les données de sentiment"""
        conn = sqlite3.connect(self.db_path)
        pd.DataFrame([data]).to_sql('sentiment_data', conn, if_exists='append', index=False)
        conn.close()

    def save_trends_data(self, data):
        """Sauvegarde les données Google Trends"""
        conn = sqlite3.connect(self.db_path)
        data.to_sql('google_trends', conn, if_exists='replace')
        conn.close()

    def load_data(self):
        # Load data into self.df
        self.df = pd.read_sql("SELECT * FROM bitcoin_data", self.connection)
    
    def some_function(self, parameter1, df):
        # Now use self.df instead of df
        result = df['column']  # Line 75 - use self.df

def train_ml_model(df: pd.DataFrame):
    # ... préparation des features et target ...
    features = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    target = (df['close'].shift(-1) > df['close']).astype(int).dropna()
    features = features.iloc[:-1]

    # Vérification
    if features.empty or target.empty or len(features) < 10:
        raise ValueError("Pas assez de données pour entraîner le modèle ML.")

    # ... suite du code ...

# 1. Ajoutez cette fonction avant d'utiliser predict_ml_signal
def predict_ml_signal(model, data):
    """
    Prédit un signal de trading avec un modèle ML
    
    Args:
        model: Modèle ML entraîné
        data: DataFrame contenant les données à prédire
    
    Returns:
        prediction: 1 (hausse) ou 0 (baisse)
        probability: Probabilité de la prédiction
    """
    # Préparer les données
    features = data[['open', 'high', 'low', 'close', 'volume']].values
    
    # Prédiction
    try:
        # Pour les modèles qui retournent directement une probabilité
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0][1]  # Probabilité de la classe 1 (hausse)
            pred = 1 if proba > 0.5 else 0
            return pred, proba
        # Pour les modèles qui ne retournent qu'une classe
        else:
            pred = model.predict(features)[0]
            return pred, 0.5  # Probabilité par défaut
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return 0, 0.0

# 2. Pour corriger le problème de df non défini, placez ce code dans une fonction
def analyze_and_predict(df):
    """
    Analyse les données et prédit le mouvement de prix
    
    Args:
        df: DataFrame avec les données à analyser
    """
    if df is not None and not df.empty and len(df) > 30:
        try:
            model = train_ml_model(df)
            last_row = df.iloc[[-1]]
            ml_pred, ml_proba = predict_ml_signal(model, last_row)
            ml_label = "Hausse probable" if ml_pred == 1 else "Baisse probable"
            st.metric("Prédiction ML", ml_label, f"Confiance : {ml_proba:.2%}")
>>>>>>> 2872012 (Initial commit: Bitcoin Analysis Dashboard)
        except Exception as e:
            st.info(f"ML non disponible : {e}")
    else:
        st.info("Pas assez de données pour entraîner le modèle ML.")