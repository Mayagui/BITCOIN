import os
import pandas as pd
import streamlit as st
import ta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import io
from data_collector import BitcoinDataCollector as DataCollector
import requests
import empyrical as ep # Assurer l'import de empyrical pour les KPIs
import base64
import sqlite3
from datetime import datetime
import joblib # Import joblib for model persistence
import xgboost as xgb # Import XGBoost

# Configuration de la page
st.set_page_config(
    page_title="Bitcoin Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- LOGO BITCOIN ANIMÉ ---
bitcoin_logo_svg = '''
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="120" height="120">
  <circle cx="32" cy="32" r="30" fill="#F7931A"/>
  <path d="M41.6,30.9c0.6-4-2.4-6.2-6.5-7.6l1.3-5.3l-3.2-0.8l-1.3,5.2c-0.8-0.2-1.7-0.4-2.5-0.6l1.3-5.2l-3.2-0.8l-1.3,5.3
  c-0.7-0.2-1.4-0.3-2-0.5l0,0l-4.4-1.1l-0.9,3.4c0,0,2.4,0.5,2.3,0.6c1.3,0.3,1.5,1.2,1.5,1.9l-1.5,6.1c0.1,0,0.2,0.1,0.3,0.1
  c-0.1,0-0.2-0.1-0.3-0.1l-2.1,8.4c-0.2,0.4-0.6,1.1-1.6,0.8c0,0.1-2.3-0.6-2.3-0.6l-1.6,3.7l4.2,1c0.8,0.2,1.5,0.4,2.3,0.6
  l-1.3,5.4l3.2,0.8l1.3-5.3c0.9,0.2,1.7,0.4,2.5,0.6l-1.3,5.3l3.2,0.8l1.3-5.4c5.4,1,9.5,0.6,11.2-4.3c1.4-3.9-0.1-6.2-2.9-7.7
  C40,35.5,41.1,33.9,41.6,30.9z M35.5,41.2c-1,3.9-7.6,1.8-9.7,1.3l1.7-7C29.7,36,36.5,37.1,35.5,41.2z M36.5,30.8
  c-0.9,3.6-6.4,1.8-8.2,1.3l1.6-6.3C31.9,26.3,37.4,27.1,36.5,30.8z" fill="white"/>
</svg>
'''

def create_animated_bitcoin_logo():
    """Crée et affiche un logo Bitcoin animé qui s'illumine"""
    st.markdown("""
    <style>
    @keyframes illuminate {
        0% { filter: drop-shadow(0 0 0px gold); }
        50% { filter: drop_shadow(0 0 15px gold); }
        100% { filter: drop_shadow(0 0 0px gold); }
    }
    .bitcoin-logo {
        animation: illuminate 2s ease-in-out infinite;
    }
    .logo-container {
        text-align: center;
        padding: 40px 0;
    }
    .app-title {
        font-size: 2.5em;
        font-weight: 700;
        color: white;
        text-shadow: 0px 0px 10px rgba(255, 215, 0, 0.5);
        margin-top: 20px;
    }
    .app-subtitle {
        font-size: 1.2em;
        color: #cccccc;
        margin-top: 5px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    b64_svg = base64.b64encode(bitcoin_logo_svg.encode()).decode()
    st.markdown(f"""
    <div class="logo-container">
        <img src="data:image/svg+xml;base64,{b64_svg}" class="bitcoin-logo" alt="Bitcoin Logo">
        <div class="app-title">Bitcoin Analytics Dashboard</div>
        <div class="app-subtitle">Analyse technique, fondamentale et machine learning</div>
    </div>
    """, unsafe_allow_html=True)


# --- CSS custom ---
def custom_css():
    st.markdown("""
    <style>
    /* Thème sombre */
    [data-theme="dark"] {
        --background-color: #0E1117;
        --text-color: #FAFAFA;
        --metric-bg: #262730;
        --metric-hover: #2E3338;
    }

    /* Thème clair - couleurs améliorées pour les graphiques */
    [data-theme="light"] .js-plotly-plot {
        --plotly-bg-color: #FFFFFF;
        --plotly-line-color: #333333;
        --plotly-text-color: #262730;
    }

    [data-theme="light"] {
        --background-color: #FFFFFF;
        --text-color: #262730;
        --metric-bg: #F0F2F6;
        --metric-hover: #E6E9EF;
    }

    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    .stMetric {
        background-color: var(--metric-bg);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stMetric:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease;
        background-color: var(--metric-hover);
    }

    .stSubheader {
        color: #00FF00;
        font-size: 1.5em;
        margin-bottom: 1em;
    }

    .stAlert {
        border-radius: 10px;
    }

    /* Style pour le sélecteur de thème */
    .theme-selector {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Database Manager ---
class DatabaseManager:
    """Gère la base SQLite pour les prix Bitcoin."""
    def __init__(self):
        self.db_path = 'bitcoin_data.db'
        self.init_database()
    
    def init_database(self):
        """Initialise la base de données SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bitcoin_prices (
                timestamp DATETIME PRIMARY KEY,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume FLOAT
            )
            ''')
            conn.commit()
            conn.close()
            st.sidebar.success("Base de données initialisée avec succès", icon="✅")
        except Exception as e:
            st.sidebar.error(f"Erreur d'initialisation de la base de données: {str(e)}", icon="❌")
    
    def save_bulk_data(self, df):
        """Sauvegarde un DataFrame entier dans la base."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Si le timestamp est l'index, le réinitialiser comme colonne
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            # S'assurer que timestamp est en format datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sélectionner uniquement les colonnes définies dans le schéma de la table
            df_to_save = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Sauvegarder dans la base de données
            df_to_save.to_sql('bitcoin_prices', conn, if_exists='append', index=False)
            conn.close()
            st.success(f"✅ {len(df_to_save)} lignes sauvegardées dans la base de données")
        except Exception as e:
            st.error(f"Erreur de sauvegarde en masse dans la DB: {str(e)}")
    
    def get_data(self, start_date=None, end_date=None):
        """Récupère les données stockées pour une période donnée."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM bitcoin_prices"
            params = []
            
            if start_date and end_date:
                query += " WHERE timestamp BETWEEN ? AND ?"
                params = [start_date, end_date]
            elif start_date:
                query += " WHERE timestamp >= ?"
                params = [start_date]
            elif end_date:
                query += " WHERE timestamp <= ?"
                params = [end_date]
            
            query += " ORDER BY timestamp ASC" # Ensure chronological order

            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
            conn.close()
            
            # Rename columns to match the application's expected OHLCV format
            df = df.rename(columns={'timestamp': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
            
            return df
        except Exception as e:
            st.error(f"Erreur de lecture depuis la DB: {str(e)}")
            return pd.DataFrame()

# --- Utilitaires ---
def rename_ohlcv_columns(df):
    # Flatten multi-level columns if they exist (common with yfinance output for multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    # Create a new mapping for robustness
    new_columns = {}
    for col in df.columns:
        # Normalize column name for comparison (lowercase, no spaces, handle common separators)
        normalized_col = col.replace('-', '_').replace(' ', '_').lower()

        if 'close' in normalized_col:
            new_columns[col] = 'close'
        elif 'high' in normalized_col:
            new_columns[col] = 'high'
        elif 'low' in normalized_col:
            new_columns[col] = 'low'
        elif 'open' in normalized_col:
            new_columns[col] = 'open'
        elif 'volume' in normalized_col:
            new_columns[col] = 'volume'
        elif 'date' in normalized_col or 'datetime' in normalized_col:
            new_columns[col] = 'timestamp'
        else:
            new_columns[col] = col # Keep original if no match

    df = df.rename(columns=new_columns)

    # Handle case where index might be 'Date' or 'Datetime' and needs to be reset
    # Ensure the index is a datetime index first for consistency with the rest of the app
    if df.index.name and df.index.name.lower() in ["date", "datetime"]:
        df = df.reset_index()
        # After reset_index, the original index name should become a column, ensure it's renamed to 'timestamp'
        df = df.rename(columns={df.index.name: 'timestamp'})
    
    # Ensure the 'timestamp' column is of datetime type if it exists as a column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']) # Drop rows where timestamp couldn't be parsed
        df = df.set_index('timestamp') # Set it as index after ensuring it's datetime

    return df

@st.cache_data(ttl=60, show_spinner="⏳ Chargement des données Bitcoin...")  # TTL réduit à 1 minute
def load_bitcoin_data():
    db = DatabaseManager()
    db.init_database()
    
    print("DEBUG: Tentative de chargement depuis la base de données locale...")
    # 1. Vérifier d'abord la base de données locale
    df = db.get_data()
    if df is not None and not df.empty:
        # Vérifier si les données sont à jour (dernière mise à jour il y a moins de 5 minutes)
        last_update = df.index.max()
        if pd.Timestamp.now() - last_update < pd.Timedelta(minutes=5):
            st.success("✅ Données locales à jour utilisées.")
            print("DEBUG: Données chargées depuis la DB et à jour.")
            return df
        else:
            print("DEBUG: Données DB non à jour, tentative de mise à jour.")
    else:
        print("DEBUG: Base de données vide ou indisponible.")
    
    print("DEBUG: Tentative de chargement depuis les fichiers CSV locaux...")
    # 2. Essayer les fichiers CSV locaux
    csv_files = ['bitcoin_data_1h.csv', 'bitcoin_data_4h.csv', 'bitcoin_4h_data.csv', 'bitcoin_1d_data.csv', 'bitcoin_backup.csv'] # Ajout de bitcoin_backup.csv
    
    for csv_file in csv_files:
        try:
            if os.path.exists(csv_file):
                temp_df = pd.read_csv(csv_file)
                temp_df = rename_ohlcv_columns(temp_df)
                
                if all(col in temp_df.columns for col in ['close', 'high', 'low', 'open', 'volume']):
                    if not isinstance(temp_df.index, pd.DatetimeIndex):
                        if 'timestamp' in temp_df.columns:
                            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce')
                            temp_df = temp_df.dropna(subset=['timestamp'])
                            temp_df = temp_df.set_index('timestamp')
                        else:
                            print(f"DEBUG: Fichier CSV {csv_file} ignoré: colonne timestamp manquante ou non valide.")
                            continue
                    
                    # Conversion optimisée des colonnes numériques
                    numeric_cols = ['close', 'high', 'low', 'open', 'volume']
                    temp_df[numeric_cols] = temp_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                    temp_df = temp_df.dropna(subset=['close'])
                    temp_df['returns'] = temp_df['close'].pct_change()
                    
                    df = temp_df.copy()
                    db.save_bulk_data(df.reset_index().copy())
                    st.success(f"🎉 Données chargées depuis {csv_file} ({len(df)} lignes).")
                    print(f"DEBUG: Données chargées depuis {csv_file}.")
                    return df
                else:
                    print(f"DEBUG: Fichier CSV {csv_file} ignoré: colonnes OHLCV manquantes.")
            else:
                print(f"DEBUG: Fichier CSV {csv_file} non trouvé.")
        except Exception as e:
            print(f"DEBUG: Erreur lors du traitement de {csv_file}: {str(e)}")
            continue
    print("DEBUG: Aucun fichier CSV local n'a pu être chargé.")

    print("DEBUG: Tentative de téléchargement depuis yfinance...")
    # 3. Si les fichiers CSV locaux échouent, essayer yfinance
    try:
        with st.spinner("⏳ Récupération des données depuis yfinance..."):
            temp_df = yf.download('BTC-USD', period='5y', interval='1d', progress=False) # Augmenter la période à 5 ans et l'intervalle à 1 jour
            if not temp_df.empty:
                temp_df = rename_ohlcv_columns(temp_df)
                temp_df['returns'] = temp_df['close'].pct_change()
                df = temp_df.copy()
                db.save_bulk_data(df.reset_index().copy())
                st.success("🎉 Données récupérées depuis yfinance avec succès.")
                print("DEBUG: Données récupérées depuis yfinance.")
                return df
            else:
                st.error("❌ yfinance n'a pas retourné de données. Vérifiez la connexion ou les paramètres.")
                print("DEBUG: yfinance a retourné un DataFrame vide.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la récupération des données depuis yfinance: {e}")
        print(f"DEBUG: Erreur yfinance: {str(e)}")
    
    st.error("❌ Impossible de charger les données Bitcoin. Veuillez réessayer.")
    print("DEBUG: Toutes les tentatives de chargement de données ont échoué.")
    return None

@st.cache_data(ttl=3600)  # Cache pour 1 heure
def fetch_fear_greed_index():
    """Récupère l'indice Fear & Greed."""
    try:
        url = "https://api.alternative.me/fng/"
        params = {
            "limit": 1,
            "format": "json"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        
        data = response.json()
        if data and 'data' in data and len(data['data']) > 0:
            score = int(data['data'][0]['value'])
            sentiment = data['data'][0]['value_classification']
            return score, sentiment
        return None, None
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ Impossible de récupérer le Fear & Greed Index: {str(e)}")
        return None, None
    except (KeyError, ValueError, IndexError) as e:
        st.warning(f"⚠️ Erreur lors du traitement du Fear & Greed Index: {str(e)}")
        return None, None

@st.cache_data(ttl=60, show_spinner="Calcul des indicateurs...")
def compute_indicators(df, rsi_window=14, bb_window=20, bb_dev=2, sma_short=20, sma_medium=50, sma_long=200, atr_window=14):
    # La vérification de la fraîcheur des données est maintenant gérée par load_bitcoin_data.
    # Si df est None ou vide à ce stade, cela devrait déjà être géré en amont dans main().

    # Calcul des indicateurs techniques
    try:
        # Moyennes mobiles
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=sma_short)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=sma_medium)
        df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=sma_long)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=rsi_window)
        
        # MACD
        df['MACD'] = ta.trend.macd(df['close'])
        df['MACD_SIGNAL'] = ta.trend.macd_signal(df['close'])
        df['MACD_HIST'] = ta.trend.macd_diff(df['close'])
        
        # Bandes de Bollinger
        df['BB_UPPER'] = ta.volatility.bollinger_hband(df['close'], window=bb_window, window_dev=bb_dev)
        df['BB_MIDDLE'] = ta.volatility.bollinger_mavg(df['close'], window=bb_window)
        df['BB_LOWER'] = ta.volatility.bollinger_lband(df['close'], window=bb_window, window_dev=bb_dev)
        
        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_window)
        
        # Volume
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # ADX (Average Directional Index)
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=atr_window)
        df['ADX_POS'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=atr_window)
        df['ADX_NEG'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=atr_window)

        # Money Flow Index (MFI)
        df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)

        # Stochastic RSI
        df['STOCH_RSI'] = ta.momentum.stochrsi(df['close'], window=rsi_window)

        # Rate of Change (ROC)
        df['ROC'] = ta.momentum.roc(df['close'], window=10)

        # Commodity Channel Index (CCI)
        df['CCI'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

        # Ultimate Oscillator
        df['Ultimate_Oscillator'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'], window1=7, window2=14, window3=28)

        # Assurez-vous que la colonne 'returns' est propre et sans NaN après sa création/calcul
        if 'returns' not in df.columns: # Vérifier si 'returns' a été calculé ailleurs (ex: load_bitcoin_data)
            df['returns'] = df['close'].pct_change()
        df['returns'] = df['returns'].fillna(0) # Remplacer les NaN ou Inf par 0
        df = df.replace([np.inf, -np.inf], 0) # Remplacer les infinis par 0

        return df
    except Exception as e:
        st.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
        return None

def compute_bollinger_bands_signal(df):
    df['BB_SIGNAL'] = 0
    df.loc[df['close'] < df['BB_LOWER'], 'BB_SIGNAL'] = 1 # Signal d'achat
    df.loc[df['close'] > df['BB_UPPER'], 'BB_SIGNAL'] = -1 # Signal de vente
    return df

def compute_rsi_signal(df):
    df['RSI_SIGNAL'] = 0
    df.loc[df['RSI'] < 30, 'RSI_SIGNAL'] = 1 # Survente
    df.loc[df['RSI'] > 70, 'RSI_SIGNAL'] = -1 # Surachat
    return df

def compute_macd_signal(df):
    df['MACD_SIGNAL_VALUE'] = 0
    df.loc[(df['MACD'] > df['MACD_SIGNAL']) & (df['MACD'].shift(1) <= df['MACD_SIGNAL'].shift(1)), 'MACD_SIGNAL_VALUE'] = 1 # Croisement haussier
    df.loc[(df['MACD'] < df['MACD_SIGNAL']) & (df['MACD'].shift(1) >= df['MACD_SIGNAL'].shift(1)), 'MACD_SIGNAL_VALUE'] = -1 # Croisement baissier
    return df

def compute_moving_average_crossover(df, short_window=50, long_window=200):
    short_ma_col = f'SMA_{short_window}'
    long_ma_col = f'SMA_{long_window}'

    if short_ma_col not in df.columns:
        df[short_ma_col] = df['close'].rolling(window=short_window).mean()
    if long_ma_col not in df.columns:
        df[long_ma_col] = df['close'].rolling(window=long_window).mean()

    df['MA_CROSSOVER_SIGNAL'] = 0
    # Croisement haussier (Golden Cross)
    df.loc[(df[short_ma_col] > df[long_ma_col]) & (df[short_ma_col].shift(1) <= df[long_ma_col].shift(1)), 'MA_CROSSOVER_SIGNAL'] = 1
    # Croisement baissier (Death Cross)
    df.loc[(df[short_ma_col] < df[long_ma_col]) & (df[short_ma_col].shift(1) >= df[long_ma_col].shift(1)), 'MA_CROSSOVER_SIGNAL'] = -1
    return df

def compute_candlestick_patterns(df):
    # Simplifié pour l'exemple, vous pouvez ajouter plus de motifs
    df['CANDLE_PATTERN_SIGNAL'] = 0
    # Exemple: Marteau haussier (petite bougie, longue mèche inférieure)
    # Assurez-vous que 'open', 'high', 'low', 'close' existent et sont numériques
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        st.warning("Colonnes OHLC manquantes pour le calcul des motifs de chandeliers.")
        return df
    
    # Exemple de Marteau haussier (Bullish Hammer)
    # Corps de bougie petit
    body = abs(df['close'] - df['open'])
    # Mèche inférieure longue (au moins 2 fois le corps)
    lower_wick = df['low'].mask(df['close'] > df['open'], df['open']) - df['low']
    lower_wick = df['low'].mask(df['close'] < df['open'], df['close']) - df['low']

    # Mèche supérieure petite (moins de 10% du corps)
    upper_wick = df['high'] - df['high'].mask(df['close'] > df['open'], df['close'])
    upper_wick = df['high'] - df['high'].mask(df['close'] < df['open'], df['open'])

    hammer_condition = (body < (df['high'] - df['low']) * 0.3) & \
                       (lower_wick > 2 * body) & \
                       (upper_wick < 0.1 * body)
    df.loc[hammer_condition, 'CANDLE_PATTERN_SIGNAL'] = 1

    return df

def compute_all_signals(df):
    df = compute_bollinger_bands_signal(df)
    df = compute_rsi_signal(df)
    df = compute_macd_signal(df)
    df = compute_moving_average_crossover(df, short_window=50, long_window=200)
    df = compute_moving_average_crossover(df, short_window=20, long_window=50) # Ajout du 20/50 MA Crossover
    df = compute_candlestick_patterns(df)
    return df

def compute_weighted_signals_score(df, fgi_score=None):
    # Initialisation des scores si les colonnes n'existent pas
    for col in ['trend_score', 'momentum_score', 'volume_score', 'volatility_score', 'final_score']:
        if col not in df.columns:
            df[col] = 0.0

    # Calcul des scores individuels
    # Trend Score (basé sur MACD, ADX, SMA Crossover)
    df['trend_score'] = 0.0
    df.loc[df['MACD_SIGNAL_VALUE'] == 1, 'trend_score'] += 0.5
    df.loc[df['MACD_SIGNAL_VALUE'] == -1, 'trend_score'] -= 0.5
    df.loc[df['ADX'] > 25, 'trend_score'] += 0.3 # Force de la tendance
    df.loc[df['MA_CROSSOVER_SIGNAL'] == 1, 'trend_score'] += 0.7 # Golden Cross
    df.loc[df['MA_CROSSOVER_SIGNAL'] == -1, 'trend_score'] -= 0.7 # Death Cross

    # Momentum Score (basé sur RSI, StochRSI, ROC)
    df['momentum_score'] = 0.0
    df.loc[df['RSI_SIGNAL'] == 1, 'momentum_score'] += 0.5
    df.loc[df['RSI_SIGNAL'] == -1, 'momentum_score'] -= 0.5
    df.loc[df['STOCH_RSI'] < 0.2, 'momentum_score'] += 0.3 # Survente StochRSI
    df.loc[df['STOCH_RSI'] > 0.8, 'momentum_score'] -= 0.3 # Surachat StochRSI
    df.loc[df['ROC'] > 0, 'momentum_score'] += 0.2 # Taux de changement positif
    df.loc[df['ROC'] < 0, 'momentum_score'] -= 0.2 # Taux de changement négatif

    # Volume Score (basé sur OBV, MFI)
    df['volume_score'] = 0.0
    # Un OBV croissant avec le prix est haussier
    df.loc[(df['OBV'].diff() > 0) & (df['close'].diff() > 0), 'volume_score'] += 0.5
    df.loc[(df['OBV'].diff() < 0) & (df['close'].diff() < 0), 'volume_score'] -= 0.5
    df.loc[df['MFI'] < 20, 'volume_score'] += 0.4 # Faible MFI (potentiel d'achat)
    df.loc[df['MFI'] > 80, 'volume_score'] -= 0.4 # Fort MFI (potentiel de vente)

    # Volatility Score (basé sur Bollinger Bands, ATR)
    df['volatility_score'] = 0.0
    df.loc[df['BB_SIGNAL'] == 1, 'volatility_score'] += 0.6 # Prix sous bande inférieure
    df.loc[df['BB_SIGNAL'] == -1, 'volatility_score'] -= 0.6 # Prix au-dessus bande supérieure
    # ATR élevé peut indiquer opportunités de breakout
    df.loc[df['ATR'] > df['ATR'].mean(), 'volatility_score'] += 0.2 # Volatilité élevée

    # Ajout de l'impact des divergences sur le score
    df.loc[df['BULLISH_DIVERGENCE'] == 1, 'trend_score'] += 0.4 # Augmenter le score de tendance pour divergence haussière
    df.loc[df['BEARISH_DIVERGENCE'] == 1, 'trend_score'] -= 0.4 # Diminuer le score de tendance pour divergence baissière

    # Score final pondéré des indicateurs techniques
    df['final_score'] = (
        df['trend_score'] * 0.35 +
        df['momentum_score'] * 0.30 +
        df['volume_score'] * 0.20 +
        df['volatility_score'] * 0.15
    )

    # Intégration du Fear & Greed Index (FGI) dans le score final
    if fgi_score is not None:
        # Normaliser le FGI de 0-100 à -1 à 1
        fgi_normalized = (fgi_score - 50) / 50
        # Appliquer un impact contrarien: la cupidité (FGI élevé) réduit le score d'achat,
        # la peur (FGI faible) augmente le score d'achat.
        # Le poids du FGI (ajustable)
        fgi_weight = 0.10 # Poids pour le FGI, vous pouvez ajuster cela
        df['final_score'] -= (fgi_normalized * fgi_weight)

    # Normalisation du score final entre -1 et 1 (si nécessaire)
    # S'assurer que df['final_score'].abs().max() n'est pas zéro pour éviter ZeroDivisionError
    if df['final_score'].abs().max() > 0:
        df['final_score'] = df['final_score'] / df['final_score'].abs().max()
    else:
        df['final_score'] = 0.0 # Ou gérer d'une autre manière si tous les scores sont zéro
            
    return df
            
def plot_price_with_signals(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, 
                        row_heights=[0.7, 0.3]) # Plus d'espace pour le graphique principal

    # Graphique principal (Prix et moyennes mobiles)
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='Prix'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', 
                             line=dict(color='#FFD700', width=1)), row=1, col=1) # Or
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', 
                             line=dict(color='#DAA520', width=1)), row=1, col=1) # Goldenrod
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200', 
                             line=dict(color='#B8860B', width=1)), row=1, col=1) # DarkGoldenrod

    # Bandes de Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='grey', width=1, dash='dash'), name='BB Supérieure'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_MIDDLE'], line=dict(color='grey', width=1, dash='dash'), name='BB Moyenne'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='grey', width=1, dash='dash'), name='BB Inférieure'), row=1, col=1)

    # Signaux d'achat/vente sur le graphique de prix (maintenant basés sur la colonne 'signal' filtrée)
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]

    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['low'] * 0.98, mode='markers', 
                                 marker=dict(symbol='triangle-up', size=10, color='green'),
                                 name='Signal Achat (Filtré)'), row=1, col=1)
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['high'] * 1.02, mode='markers', 
                                 marker=dict(symbol='triangle-down', size=10, color='red'),
                                 name='Signal Vente (Filtré)'), row=1, col=1)
    
    # Nouveau: Affichage des divergences
    bullish_divergences = df[df['BULLISH_DIVERGENCE'] == 1]
    bearish_divergences = df[df['BEARISH_DIVERGENCE'] == 1]

    if not bullish_divergences.empty:
        fig.add_trace(go.Scatter(x=bullish_divergences.index, y=bullish_divergences['low'] * 0.95, mode='markers', 
                                 marker=dict(symbol='star', size=12, color='purple'),
                                 name='Divergence Haussière'), row=1, col=1)
    if not bearish_divergences.empty:
        fig.add_trace(go.Scatter(x=bearish_divergences.index, y=bearish_divergences['high'] * 1.05, mode='markers', 
                                 marker=dict(symbol='star', size=12, color='orange'),
                                 name='Divergence Baissière'), row=1, col=1)

    # Graphique du score final
    fig.add_trace(go.Scatter(x=df.index, y=df['final_score'], mode='lines', name='Score Final', 
                             line=dict(color='blue')), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1) # Ligne de référence à 0
    fig.add_hrect(y0=-0.5, y1=0.5, line_width=0, fillcolor="rgba(128,128,128,0.2)", layer="below", row=2, col=1) # Zone neutre

    fig.update_layout(title_text='Analyse du Prix et des Signaux', height=700, 
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text='Prix', row=1, col=1)
    fig.update_yaxes(title_text='Score', row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

def plot_macd(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', 
                             line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], mode='lines', name='Signal', 
                             line=dict(color='red')), row=1, col=1)
    # Histogramme MACD
    colors = ['green'] * len(df)
    colors = ['red' if val < 0 else 'green' for val in df['MACD_HIST']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_HIST'], name='Histogramme', 
                         marker_color=colors), row=2, col=1)

    fig.update_layout(title_text='Indicateur MACD', height=500, 
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text='MACD', row=1, col=1)
    fig.update_yaxes(title_text='Histogramme', row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="Surachat (70)", 
                  annotation_position="top right")
    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                  annotation_text="Survente (30)", 
                  annotation_position="bottom right")
    fig.update_layout(title_text='Indicateur RSI', height=400,
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_bollinger_bands(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='Prix'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='blue', width=1), name='Bande Supérieure'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_MIDDLE'], line=dict(color='red', width=1), name='Moyenne Mobile'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='blue', width=1), name='Bande Inférieure'))
    fig.update_layout(title_text='Bandes de Bollinger', height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Nouvelle fonction pour les signaux de détection de divergence ---
def detect_divergences(df, price_col, indicator_col, window=14):
    # Check if df is None or empty
    if df is None or df.empty:
        st.warning("DataFrame est None ou vide. Impossible de détecter les divergences.")
        return pd.DataFrame()  # Retourne un DataFrame vide au lieu de None

    # Assurez-vous que les colonnes nécessaires existent
    if price_col not in df.columns or indicator_col not in df.columns:
        st.warning(f"Colonnes manquantes pour la détection de divergence: {price_col} ou {indicator_col}")
        # Pour assurer que les colonnes de divergence existent pour les étapes suivantes, nous les initialisons ici.
        df['BULLISH_DIVERGENCE'] = 0
        df['BEARISH_DIVERGENCE'] = 0
        return df # Retourne le df avec les colonnes de divergence initialisées

    df['BULLISH_DIVERGENCE'] = 0
    df['BEARISH_DIVERGENCE'] = 0

    # Détection des divergences haussières (prix plus bas, indicateur plus haut)
    # Nous cherchons un plus bas significatif dans le prix
    lows_price = df[price_col].rolling(window=window).min()
    # Et un plus haut significatif dans l'indicateur (qui ne correspond pas au prix)
    highs_indicator = df[indicator_col].rolling(window=window).max()

    for i in range(window, len(df)):
        # Divergence haussière: prix fait un plus bas, mais l'indicateur fait un plus haut ou un plus bas moins bas
        if df[price_col].iloc[i] < df[price_col].iloc[i-window] and \
           df[indicator_col].iloc[i] > df[indicator_col].iloc[i-window]:
            df.loc[df.index[i], 'BULLISH_DIVERGENCE'] = 1
        
        # Divergence baissière: prix fait un plus haut, mais l'indicateur fait un plus bas ou un plus haut moins haut
        elif df[price_col].iloc[i] > df[price_col].iloc[i-window] and \
             df[indicator_col].iloc[i] < df[indicator_col].iloc[i-window]:
            df.loc[df.index[i], 'BEARISH_DIVERGENCE'] = 1

    return df

def display_kpis(df, fgi_score=None, fgi_sentiment=None):
    # Calcul des KPIs
    total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    # Check if returns are available and not all NaN before calculating annualized_return, max_dd, sharpe, sortino
    if 'returns' in df.columns and not df['returns'].dropna().empty:
        annualized_return = ep.annual_return(df['returns'])
        max_dd = ep.max_drawdown(df['returns'])
        try:
            sharpe = ep.sharpe_ratio(df['returns'])
        except ZeroDivisionError:
            sharpe = 0.0 # Handle case where risk is zero
        try:
            sortino = ep.sortino_ratio(df['returns'])
        except ZeroDivisionError:
            sortino = 0.0 # Handle case where risk is zero
    else:
        annualized_return = 0.0
        max_dd = 0.0
        sharpe = 0.0
        sortino = 0.0


    st.markdown("### Indicateurs Clés de Performance (KPIs) du Prix Brut")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label="📈 Rendement Total", value=f"{total_return:.2%}", delta=f"{total_return:.2%}")
    col2.metric(label="📊 Rendement Annualisé", value=f"{annualized_return:.2%}", delta=f"{annualized_return:.2%}")
    col3.metric(label="📉 Max Drawdown", value=f"{max_dd:.2%}", delta=f"{max_dd:.2%}")
    col4.metric(label="⭐ Ratio de Sharpe", value=f"{sharpe:.2f}")
    col5.metric(label="🏆 Ratio de Sortino", value=f"{sortino:.2f}")
    
    if fgi_score is not None and fgi_sentiment is not None:
        st.markdown("### 🧠 Fear & Greed Index (FGI)")
        st.info(f"Le **Fear & Greed Index** actuel est de **{fgi_score}** (Sentiment: **{fgi_sentiment}**). Ce score indique le sentiment général du marché : une valeur faible indique la peur, une valeur élevée indique la cupidité.")

def display_candlestick_chart(df):
    st.subheader("Graphique en Chandeliers")
    fig = go.Figure(data=[go.Candlestick(x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
                                        close=df['close'])])
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def display_metrics(df):
    st.subheader("Tableau des Métriques")
    st.write(df.tail())

def display_signal_strategy_section(df, signal_threshold):
    st.subheader("Stratégie de Signaux")
    # signal_threshold est maintenant passé en argument depuis la barre latérale

    df['signal'] = 0 # 1 pour achat, -1 pour vente, 0 pour neutre
    df.loc[df['final_score'] >= signal_threshold, 'signal'] = 1
    df.loc[df['final_score'] <= -signal_threshold, 'signal'] = -1

    buy_signals_count = df[df['signal'] == 1].shape[0]
    sell_signals_count = df[df['signal'] == -1].shape[0]
    st.info(f"Nombre de signaux d'achat générés: {buy_signals_count}")
    st.info(f"Nombre de signaux de vente générés: {sell_signals_count}")

    st.write("Signaux générés (derniers 10) :")
    st.dataframe(df[df['signal'] != 0][['close', 'final_score', 'signal']].tail(10))

    st.write("### Alertes de Trading Basées sur le Score Global")
    if df['final_score'].iloc[-1] > 0.7:
        st.success("🟢 **Signal d'Achat Fort :** Le score global indique une forte opportunité d'entrée. Considérez l'achat.")
    elif df['final_score'].iloc[-1] > 0.3:
        st.info("🔵 **Signal d'Achat Modéré :** Le score global suggère une opportunité d'achat. Surveillez le marché.")
    elif df['final_score'].iloc[-1] < -0.7:
        st.error("🔴 **Signal de Vente Fort :** Le score global indique une forte opportunité de sortie/vente. Considérez la vente.")
    elif df['final_score'].iloc[-1] < -0.3:
        st.warning("🟠 **Signal de Vente Modéré :** Le score global suggère une opportunité de vente. Surveillez le marché.")
    else:
        st.caption("⚪️ **Marché Neutre :** Le score global ne donne pas de signal clair. Attendez une direction.")

def filter_signals_by_volatility(df, signal_column='signal', atr_threshold_percentile=0.2):
    # Calcul du seuil d'ATR basé sur le percentile
    if not df['ATR'].empty:
        atr_threshold = df['ATR'].quantile(atr_threshold_percentile)
    else:
        atr_threshold = 0  # Fallback if ATR is empty

    df.loc[df['ATR'] < atr_threshold, signal_column] = 0
    return df

# --- Machine Learning Section ---
def prepare_data_for_ml(df):
    # Préparer les caractéristiques (features) et la cible (target)
    # Supposons que notre cible est la direction du mouvement du prix (haut/bas)
    df['price_direction'] = (df['close'].shift(-1) > df['close']).astype(int) # 1 si le prix monte le jour suivant, 0 sinon

    features = [
        'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'STOCH_RSI',
        'OBV', 'MFI', 'ATR', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',
        'SMA_20', 'SMA_50', 'SMA_200', 'ADX', 'CCI', 'ROC', 'Ultimate_Oscillator',
        'trend_score', 'momentum_score', 'volume_score', 'volatility_score', 'final_score'
    ]
    # Supprimer les lignes avec des NaN qui résulteraient des calculs d'indicateurs
    df_ml = df[features + ['price_direction']].dropna()

    X = df_ml[features]
    y = df_ml['price_direction']

    # Normalisation des caractéristiques (sera effectuée DANS train_ml_model pour éviter le data leakage)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    return X, y, df_ml, features # Simplified return to provide X (unscaled), y, the processed df, and feature names

def train_ml_model(X, y, model_type):
    # Assurer que les données ne sont pas mélangées aléatoirement pour les séries temporelles
    # Le test_size de 0.2 signifie que les 20% de données les plus récentes seront pour le test.
    split_index = int(len(X) * 0.8)
    X_train_unscaled, X_test_unscaled = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Normalisation des caractéristiques: fitter le scaler uniquement sur l'ensemble d'entraînement
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_unscaled)
    X_test = scaler.transform(X_test_unscaled)

    print(f"DEBUG: Class distribution in y_train:\n{y_train.value_counts(normalize=True)}")
    print(f"DEBUG: Class distribution in y_test:\n{y_test.value_counts(normalize=True)}")

    model = None
    y_pred_proba = None

    if model_type == 'XGBoost':
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42, solver='liblinear')
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        st.error("Type de modèle non reconnu.")
        return None, 0.0, np.array([[0,0],[0,0]]), None, 0.0, 0.0, 0.0, 0.0 # Return default values

    # Train the model if it's not None
    if model is not None:
        model.fit(X_train, y_train)

    # Make predictions and calculate metrics
    y_pred = None
    if model is not None:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

    if y_pred is None:
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        roc_auc = 0.0
        cm = np.array([[0,0],[0,0]])
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba) # Use probabilities for ROC AUC
        except ValueError:
            roc_auc = 0.0
        cm = confusion_matrix(y_test, y_pred)

    return model, accuracy, cm, precision, recall, f1, roc_auc, scaler

def load_ml_model(model_type):
    """
    Charge un modèle ML sauvegardé et son scaler.
    """
    model_filename = f"ml_model_{model_type.lower()}.pkl"
    scaler_filename = f"ml_scaler_{model_type.lower()}.pkl"
    model = None
    scaler = None
    try:
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        st.success(f"Modèle {model_filename} et scaler {scaler_filename} chargés avec succès.")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Erreur: Les fichiers modèle ({model_filename}) ou scaler ({scaler_filename}) n'ont pas été trouvés pour {model_type}. Veuillez entraîner un modèle d'abord.")
        return None, None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle ou du scaler {model_type}: {e}")
        return None, None

def predict_next_day_price(model, last_data_point_scaled):
    """
    Prédit la direction du prix pour le prochain jour.
    last_data_point_scaled doit être une seule ligne normalisée (X_test d'une seule observation).
    """
    if model is None:
        return 0, 0.0 # Return 0 (baisse) and 0% confidence if no model

    # For scikit-learn classifiers, predict_proba returns [prob_class_0, prob_class_1]
    predictions_proba = model.predict_proba(last_data_point_scaled.reshape(1, -1))[0]
    pred = int(model.predict(last_data_point_scaled.reshape(1, -1))[0])
    confidence = float(predictions_proba[pred])

    return pred, confidence

def make_ml_predictions(model, X_data, df_original):
    """
    Fait des prédictions avec le modèle entraîné et ajoute les résultats au DataFrame original.
    X_data can be X_test (for evaluation) or X_full_scaled (for historical visualization).
    """
    predictions_raw = model.predict_proba(X_data)[:, 1] # Get probability of class 1

    predictions = (predictions_raw > 0.5).astype(int) # Class 1 (increase) if prob > 0.5
    confidence_scores = predictions_raw # Probability of the positive class

    # Ensure predictions match length of original DataFrame
    if len(predictions) != len(df_original):
        st.warning("Inconsistency in length between predictions and original DataFrame. Some visualizations might be affected.")
        # Attempt to truncate or align if possible, or return empty DataFrame
        # For this use, we expect X_data to be X_full_scaled, so length should match df_original.
        # If X_data is X_test (split), then df_original should not be the full df.
        # To simplify, we will assume this function receives the correct length and an aligned df_original.

    # Add predictions and confidence to the original DataFrame
    df_result = df_original.copy()
    df_result['predicted_direction'] = predictions
    df_result['prediction_confidence'] = confidence_scores
    
    return df_result

def display_ml_section(df):
    st.header("🔮 Prédictions Machine Learning")
    
    # Préparation des données (X est maintenant non-normalisé ici)
    X_unscaled, y, df_ml_original, features_names = prepare_data_for_ml(df)
    
    if X_unscaled.shape[0] == 0:
        st.warning("Pas assez de données pour l'entraînement du modèle après la suppression des NaN.")
        return

    # Sélection du modèle
    model_options = ['XGBoost', 'LogisticRegression', 'RandomForest'] # Simplified to 3 models
    model_type = st.selectbox("Choisir le type de modèle", model_options, index=0)

    # Bouton pour entraîner le modèle
    if st.button(f"Entraîner et Sauvegarder le modèle {model_type}"):
        with st.spinner(f"Entraînement du modèle {model_type} en cours..."):
            # Pass the scaler for saving
            model, accuracy, cm, precision, recall, f1, roc_auc, scaler = train_ml_model(X_unscaled, y, model_type)
            if model is not None:
                joblib.dump(model, f"ml_model_{model_type.lower()}.pkl") # Save the trained model
                joblib.dump(scaler, f"ml_scaler_{model_type.lower()}.pkl") # Save the scaler
                st.success("Modèle et scaler sauvegardés avec succès !")

                st.write(f"Précision (Accuracy) du modèle {model_type} sur l'ensemble de test: {accuracy:.2%}")
                st.write(f"Précision (Precision): {precision:.2%}")
                st.write(f"Rappel (Recall): {recall:.2%}")
                st.write(f"Score F1: {f1:.2f}")
                st.write(f"AUC ROC: {roc_auc:.2f}")
                st.write("Matrice de Confusion (sur l'ensemble de test):")
                fig_cm, ax_cm = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
                disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
                st.pyplot(fig_cm)
            else:
                st.error("L'entraînement du modèle a échoué.")

    st.markdown("--- Prédictions avec le modèle sauvegardé ---") # Changed title for clarity

    # Button to predict next trend
    if st.button(f"Prédire la tendance du prochain jour avec le modèle {model_type}"):
        model, loaded_scaler = load_ml_model(model_type)
        if model and loaded_scaler:
            # Prepare the last row of UNNORMALIZED data for prediction
            # Ensure columns of X_original_unscaled match features used for training
            last_data_point_unscaled = X_unscaled.tail(1)
            last_data_point_scaled = loaded_scaler.transform(last_data_point_unscaled) # Normalize with the loaded scaler
            
            pred, confidence = predict_next_day_price(model, last_data_point_scaled) # features_names is no longer required here

            st.write("### Prédiction pour le prochain jour")
            if pred == 1:
                st.markdown(f"## <span style='color:limegreen'>Tendance : Hausse probable</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"## <span style='color:#FF4B4B'>Tendance : Baisse probable</span>", unsafe_allow_html=True)
            st.markdown(f"**Confiance :** <span style='font-weight:bold;'>{confidence:.2%}</span>", unsafe_allow_html=True)
        else:
            st.warning("Modèle ou scaler non chargé. Entraînez et sauvegardez le modèle d'abord.")

    # Visualization of past predictions (optional, based on make_ml_predictions)
    st.info("Les prédictions passées et l'analyse de leur performance sont uniquement pour la visualisation et n'impliquent pas de backtesting de stratégie de trading.")
    if st.checkbox("Afficher les prédictions passées"):
        model, loaded_scaler = load_ml_model(model_type)
        if model and loaded_scaler:
            # Normalize ALL data from X_unscaled (unnormalized features)
            X_full_scaled = loaded_scaler.transform(X_unscaled)
            full_predictions_df = make_ml_predictions(model, X_full_scaled, df_ml_original) # df_ml_original est le df nettoyé avec price_direction

            st.write("### Prédictions de direction de prix (derniers 10) :")
            # Ensure 'price_direction' still exists in full_predictions_df (it should be there via df_ml_original)
            st.dataframe(full_predictions_df[['close', 'price_direction', 'predicted_direction', 'prediction_confidence']].tail(10))

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=full_predictions_df.index, y=full_predictions_df['close'], mode='lines', name='Prix'))
            
            # Filter signals with a minimum confidence (example: 60%)
            confidence_threshold = st.slider("Seuil de confiance pour les prédictions ML", min_value=0.5, max_value=1.0, value=0.6, step=0.05)

            buy_preds = full_predictions_df[(full_predictions_df['predicted_direction'] == 1) & (full_predictions_df['prediction_confidence'] >= confidence_threshold)]
            sell_preds = full_predictions_df[(full_predictions_df['predicted_direction'] == 0) & (full_predictions_df['prediction_confidence'] >= confidence_threshold)]

            if not buy_preds.empty:
                fig_pred.add_trace(go.Scatter(x=buy_preds.index, y=buy_preds['close'] * 0.98, mode='markers',
                                             marker=dict(symbol='triangle-up', size=10, color='green'),
                                             name='Prédiction Achat (ML, Confiance filtrée)'))
            if not sell_preds.empty:
                fig_pred.add_trace(go.Scatter(x=sell_preds.index, y=sell_preds['close'] * 1.02, mode='markers',
                                             marker=dict(symbol='triangle-down', size=10, color='red'),
                                             name='Prédiction Vente (ML, Confiance filtrée)'))

            fig_pred.update_layout(title_text='Prix et Prédictions du Modèle ML (Filtrées par Confiance)', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning("Impossible d'afficher les prédictions passées : modèle ou scaler non chargé.")

def run_advanced_backtest(df, initial_capital=100000, start_date=None, end_date=None):
    """
    Exécute un backtest avancé comparant 4 stratégies différentes.
    
    Stratégies:
    1. Buy & Hold (référence)
    2. Stratégie basée sur les signaux techniques (RSI + MACD + BB)
    3. Stratégie basée sur le score final pondéré
    4. Stratégie basée sur les divergences
    """
    st.header("📊 Backtest Avancé des Stratégies")
    
    # Configuration du backtest
    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Capital Initial (€)", value=100000, step=10000)
    with col2:
        strategy_period = st.selectbox(
            "Période de Backtest",
            ["1 mois", "3 mois", "6 mois", "1 an", "2 ans", "3 ans", "5 ans", "Tout"],
            index=3
        )
    
    # Filtrer les données selon la période sélectionnée
    if strategy_period != "Tout":
        periods = {
            "1 mois": "30D",
            "3 mois": "90D",
            "6 mois": "180D",
            "1 an": "365D",
            "2 ans": "730D",
            "3 ans": "1095D",
            "5 ans": "1825D"
        }
        df = df.last(periods[strategy_period])
    
    # Initialisation des DataFrames pour chaque stratégie
    strategies = {
        'Buy & Hold': pd.DataFrame(index=df.index),
        'Technique': pd.DataFrame(index=df.index),
        'Score Pondéré': pd.DataFrame(index=df.index),
        'Divergences': pd.DataFrame(index=df.index)
    }
    
    # 1. Stratégie Buy & Hold
    strategies['Buy & Hold']['position'] = 1  # Toujours en position
    strategies['Buy & Hold']['returns'] = df['returns']
    
    # 2. Stratégie Technique (RSI + MACD + BB)
    strategies['Technique']['position'] = 0
    # Signal d'achat si RSI < 30 ou MACD croisement haussier ou prix < BB inférieure
    strategies['Technique'].loc[
        (df['RSI_SIGNAL'] == 1) | 
        (df['MACD_SIGNAL_VALUE'] == 1) | 
        (df['BB_SIGNAL'] == 1), 
        'position'
    ] = 1
    # Signal de vente si RSI > 70 ou MACD croisement baissier ou prix > BB supérieure
    strategies['Technique'].loc[
        (df['RSI_SIGNAL'] == -1) | 
        (df['MACD_SIGNAL_VALUE'] == -1) | 
        (df['BB_SIGNAL'] == -1), 
        'position'
    ] = -1
    strategies['Technique']['returns'] = strategies['Technique']['position'].shift(1) * df['returns']
    
    # 3. Stratégie Score Pondéré
    strategies['Score Pondéré']['position'] = 0
    strategies['Score Pondéré'].loc[df['final_score'] >= 0.5, 'position'] = 1
    strategies['Score Pondéré'].loc[df['final_score'] <= -0.5, 'position'] = -1
    strategies['Score Pondéré']['returns'] = strategies['Score Pondéré']['position'].shift(1) * df['returns']
    
    # 4. Stratégie Divergences
    strategies['Divergences']['position'] = 0
    strategies['Divergences'].loc[df['BULLISH_DIVERGENCE'] == 1, 'position'] = 1
    strategies['Divergences'].loc[df['BEARISH_DIVERGENCE'] == 1, 'position'] = -1
    strategies['Divergences']['returns'] = strategies['Divergences']['position'].shift(1) * df['returns']
    
    # Calcul des métriques de performance pour chaque stratégie
    performance_metrics = {}
    for strategy_name, strategy_df in strategies.items():
        # Calcul des rendements cumulatifs
        strategy_df['cumulative_returns'] = (1 + strategy_df['returns']).cumprod() - 1
        
        # Calcul de la valeur du portefeuille
        strategy_df['portfolio_value'] = initial_capital * (1 + strategy_df['cumulative_returns'])
        
        # Calcul des métriques
        total_return = strategy_df['cumulative_returns'].iloc[-1]
        annualized_return = ((1 + total_return) ** (252 / len(strategy_df)) - 1)
        sharpe_ratio = np.sqrt(252) * strategy_df['returns'].mean() / strategy_df['returns'].std()
        max_drawdown = (strategy_df['portfolio_value'] / strategy_df['portfolio_value'].cummax() - 1).min()
        
        performance_metrics[strategy_name] = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Portfolio Value': strategy_df['portfolio_value'].iloc[-1]
        }
    
    # Affichage des résultats
    st.subheader("📈 Performance des Stratégies")
    
    # Graphique de comparaison des valeurs de portefeuille
    fig = go.Figure()
    for strategy_name, strategy_df in strategies.items():
        fig.add_trace(go.Scatter(
            x=strategy_df.index,
            y=strategy_df['portfolio_value'],
            name=strategy_name,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Évolution de la Valeur du Portefeuille par Stratégie',
        xaxis_title='Date',
        yaxis_title='Valeur (€)',
        template='plotly_dark',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des métriques de performance
    metrics_df = pd.DataFrame(performance_metrics).T
    metrics_df = metrics_df.round(4)
    metrics_df['Total Return'] = metrics_df['Total Return'].map('{:.2%}'.format)
    metrics_df['Annualized Return'] = metrics_df['Annualized Return'].map('{:.2%}'.format)
    metrics_df['Max Drawdown'] = metrics_df['Max Drawdown'].map('{:.2%}'.format)
    metrics_df['Portfolio Value'] = metrics_df['Portfolio Value'].map('{:,.2f} €'.format)
    
    st.subheader("📊 Métriques de Performance")
    st.dataframe(metrics_df)
    
    # Analyse des transactions
    st.subheader("🔄 Analyse des Transactions")
    for strategy_name, strategy_df in strategies.items():
        if strategy_name != 'Buy & Hold':
            trades = strategy_df[strategy_df['position'] != strategy_df['position'].shift(1)]
            buy_trades = trades[trades['position'] == 1]
            sell_trades = trades[trades['position'] == -1]
            
            st.write(f"**{strategy_name}**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre d'achats", len(buy_trades))
            with col2:
                st.metric("Nombre de ventes", len(sell_trades))
    
    return strategies, performance_metrics

# --- Main Application Logic ---
def main():
    st.title("Bitcoin Analytics Dashboard")
    create_animated_bitcoin_logo()
    custom_css()

    # Delete existing database file to ensure fresh start and correct schema
    if os.path.exists('bitcoin_data.db'):
        try:
            os.remove('bitcoin_data.db')
            st.sidebar.info("🗑️ Ancienne base de données supprimée pour une réinitialisation propre. Cela assure que vous travaillez avec les données les plus récentes.")
        except Exception as e:
            st.sidebar.error(f"❌ Impossible de supprimer l'ancienne base de données: {e}. Veuillez vérifier les permissions.")
    
    # Fetch Fear & Greed Index
    fgi_score, fgi_sentiment = fetch_fear_greed_index()
    if fgi_score is None:
        st.sidebar.warning("⚠️ Impossible de récupérer le Fear & Greed Index. L'analyse ne l'inclura pas pour cette session.")

    # Widgets de configuration des indicateurs dans la barre latérale
    with st.sidebar:
        st.header("⚙️ Paramètres de l'Application")

        # New: Price Evolution Period
        price_evolution_period = st.selectbox(
            "Période d'évolution du prix Bitcoin",
            ("7 jours", "1 mois", "3 mois", "6 mois", "12 mois", "2 ans", "3 ans", "4 ans", "5 ans"),
            index=4, # Default to 12 months
            help="Sélectionnez la période pour afficher l'évolution du prix du Bitcoin."
        )

        with st.expander("📊 Configuration des Indicateurs Techniques", expanded=True):
            st.markdown("Réglez les périodes pour les calculs des indicateurs (RSI, Bandes de Bollinger, Moyennes Mobiles, ATR). Les indicateurs affichés sont les SMA, RSI, MACD et Bandes de Bollinger.")
            rsi_window = st.slider("Période RSI", min_value=7, max_value=30, value=14, step=1, help="Fenêtre de calcul pour le Relative Strength Index.")
            bb_window = st.slider("Période Bandes de Bollinger", min_value=10, max_value=50, value=20, step=1, help="Nombre de jours pour le calcul des Bandes de Bollinger.")
            bb_dev = st.slider("Écart-type Bandes de Bollinger", min_value=1.0, max_value=3.0, value=2.0, step=0.1, help="Nombre d'écarts-types pour les bandes supérieure et inférieure.")
            sma_short = st.slider("Période SMA Courte", min_value=5, max_value=50, value=20, step=1, help="Fenêtre pour la moyenne mobile simple courte.")
            sma_medium = st.slider("Période SMA Moyenne", min_value=20, max_value=100, value=50, step=1, help="Fenêtre pour la moyenne mobile simple moyenne.")
            sma_long = st.slider("Période SMA Longue", min_value=50, max_value=300, value=200, step=5, help="Fenêtre pour la moyenne mobile simple longue.")
            atr_window = st.slider("Période ATR", min_value=7, max_value=30, value=14, step=1, help="Fenêtre pour l'Average True Range (mesure de volatilité). Cet indicateur est utilisé en interne pour le filtrage mais n'est plus affiché pour simplifier.")
        
        with st.expander("🔍 Filtrage des Signaux", expanded=True):
            st.markdown("Ajustez ce seuil pour filtrer les signaux basés sur la volatilité du marché.")
            atr_threshold_percentile = st.slider("Seuil de Volatilité (Percentile ATR)", min_value=0.0, max_value=0.5, value=0.1, step=0.01,
                                             help="Les signaux sont désactivés si l'ATR est en dessous de ce percentile. Une valeur plus élevée filtre plus agressivement les signaux en période de faible volatilité.")
            signal_threshold = st.slider("Seuil de signalisation pour le score final", min_value=-1.0, max_value=1.0, value=0.5, step=0.1)
        
    df = load_bitcoin_data()

    if df is None or df.empty:
        st.error("❌ Erreur critique : Impossible de charger les données Bitcoin. Veuillez vérifier votre connexion Internet et réessayer.")
        return # Stop execution if data is not loaded

    # Adjust DataFrame based on selected price evolution period
    if price_evolution_period == "7 jours":
        df = df.last('7D')
    elif price_evolution_period == "1 mois":
        df = df.last('30D')
    elif price_evolution_period == "3 mois":
        df = df.last('90D')
    elif price_evolution_period == "6 mois":
        df = df.last('180D')
    elif price_evolution_period == "12 mois":
        df = df.last('365D')
    elif price_evolution_period == "2 ans":
        df = df.last('730D')
    elif price_evolution_period == "3 ans":
        df = df.last('1095D')
    elif price_evolution_period == "4 ans":
        df = df.last('1460D')
    elif price_evolution_period == "5 ans":
        df = df.last('1825D')
    
    if not df.empty:
        df = compute_indicators(df, rsi_window=rsi_window, bb_window=bb_window, bb_dev=bb_dev, 
                                sma_short=sma_short, sma_medium=sma_medium, sma_long=sma_long, atr_window=atr_window)
        df = compute_all_signals(df)
        
        # Ensure divergences are calculated before compute_weighted_signals_score
        df = detect_divergences(df, 'close', 'RSI')
        df = detect_divergences(df, 'close', 'MACD')
        
        df = compute_weighted_signals_score(df, fgi_score=fgi_score)
        
        # Define the main signal and apply volatility filtering here
        df['signal'] = 0  # 1 for buy, -1 for sell, 0 for neutral
        df.loc[df['final_score'] >= signal_threshold, 'signal'] = 1
        df.loc[df['final_score'] <= -signal_threshold, 'signal'] = -1
        df = filter_signals_by_volatility(df, signal_column='signal', atr_threshold_percentile=atr_threshold_percentile)

        tab1, tab2, tab3, tab4 = st.tabs(["Vue d'Ensemble", "Analyse Technique", "Backtest", "Machine Learning"])

        with tab1:
            st.subheader("📈 Indicateurs Clés de Performance (KPIs)")
            display_kpis(df, fgi_score, fgi_sentiment)
            st.subheader("🕯️ Graphique en Chandeliers Historique")
            display_candlestick_chart(df)
            st.subheader("📋 Tableau des Métriques et Données Récentes")
            display_metrics(df)
            st.subheader("🚦 Stratégie de Signaux Globaux")
            display_signal_strategy_section(df, signal_threshold)

        with tab2:
            st.subheader("📊 Graphique Prix & Signaux")
            plot_price_with_signals(df)
            st.subheader("📉 Indicateur MACD")
            plot_macd(df)
            st.subheader("⚡ Indicateur RSI")
            plot_rsi(df)
            st.subheader("〰️ Bandes de Bollinger")
            plot_bollinger_bands(df)
            
        with tab3:
            st.subheader("🧪 Backtest des Stratégies")
            strategies, performance_metrics = run_advanced_backtest(df)

        with tab4:
            st.subheader("🤖 Prédictions Machine Learning")
            display_ml_section(df)
            
if __name__ == "__main__":
    main()
