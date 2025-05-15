import sqlite3
import pandas as pd
import streamlit as st

class DatabaseManager:
    def __init__(self, db_path='bitcoin_data.db'):
        self.db_path = db_path
        self.init_database()
    
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

def train_ml_model(df: pd.DataFrame):
    # ... préparation des features et target ...
    features = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    target = (df['close'].shift(-1) > df['close']).astype(int).dropna()
    features = features.iloc[:-1]

    # Vérification
    if features.empty or target.empty or len(features) < 10:
        raise ValueError("Pas assez de données pour entraîner le modèle ML.")

    # ... suite du code ...

# ...avant d'appeler train_ml_model(df)...
if df is not None and not df.empty and len(df) > 30:
    try:
        model = train_ml_model(df)
        last_row = df.iloc[[-1]]
        ml_pred, ml_proba = predict_ml_signal(model, last_row)
        ml_label = "Hausse probable" if ml_pred == 1 else "Baisse probable"
        st.metric("Prédiction ML", ml_label, f"Confiance : {ml_proba:.2%}")
    except Exception as e:
        st.info(f"ML non disponible : {e}")
else:
    st.info("Pas assez de données pour entraîner le modèle ML.")