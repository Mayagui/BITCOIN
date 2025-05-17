import os
import pandas as pd
import streamlit as st
import ta
import yfinance as yf
import matplotlib.pyplot as plt

# Téléchargement des données si le fichier n'existe pas
if not os.path.exists("bitcoin_5y.csv"):
    df = yf.download("BTC-USD", period="5y", interval="1d")
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Datetime", "Close": "close", "High": "high", "Low": "low", "Open": "open", "Volume": "volume"}, inplace=True)
    # Si la colonne Datetime n'existe pas correctement, on la recrée
    if "Datetime" not in df.columns or df["Datetime"].isnull().all():
        df['Datetime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    df = df[["Datetime", "close", "high", "low", "open", "volume"]]
    df.to_csv("bitcoin_5y.csv", index=False)

# Lecture du CSV
df = pd.read_csv('bitcoin_5y.csv')

# Vérification ou création de la colonne Datetime
if 'Datetime' not in df.columns:
    df['Datetime'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

# Conversion en datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

# S'assurer que l'index est bien un DatetimeIndex et sans NaT
if not isinstance(df.index, pd.DatetimeIndex):
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.set_index('Datetime')
    else:
        df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
df = df[~df.index.isna()]
if not isinstance(df.index, pd.DatetimeIndex) or df.index.isnull().any() or len(df) == 0:
    st.error("Impossible de créer un index de dates valide pour le DataFrame.")
    st.stop()

# Exemple : filtrer les 30 derniers jours
nb_jours = 30
last_date = df.index.max()
date_min = last_date - pd.Timedelta(days=nb_jours)
df_periode = df[df.index >= date_min]

# Afficher le nombre de lignes filtrées
st.write(f"Nombre de lignes sur les {nb_jours} derniers jours :", len(df_periode))

# Conversion en numérique pour les colonnes concernées
for col in ['close', 'high', 'low', 'open', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculs d'indicateurs de base
df["returns"] = df["close"].pct_change()
df["volatility"] = df["returns"].rolling(window=30).std() * (365 ** 0.5)
df["SMA_50"] = df["close"].rolling(window=50).mean()
df["SMA_200"] = df["close"].rolling(window=200).mean()

cross = (df["SMA_50"] > df["SMA_200"]) & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
df.loc[cross, "signal"] = "golden_cross"
df["trend"] = df["close"].diff().apply(lambda x: "up" if x > 0 else "down")

zero_vol = (df['volume'] == 0).sum()
if zero_vol > 0:
    st.warning(f"{zero_vol} lignes ont un volume nul (0).")

st.write(df.head())
st.write("close dtype:", df['close'].dtype)
st.write("close NaN count:", df['close'].isna().sum())
st.write("Première date du DataFrame :", df.index.min())
st.write("Dernière date du DataFrame :", df.index.max())
st.write("Nombre de lignes dans le DataFrame :", len(df))

# Filtrage des données par période grâce à la sidebar
start_date = st.date_input("Début", value=df.index.min().date())
end_date = st.date_input("Fin", value=df.index.max().date())
df_filtered = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

# Style CSS personnalisé
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #FFFFFF; }
    .stMetric { background-color: #1E2130; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .stMetric:hover { transform: translateY(-5px); transition: all 0.3s ease;}
    .indicator-box { background-color: #1E2130; padding: 15px; border-radius: 8px; margin: 10px 0;}
    .signal-buy { color: #00FF00; font-weight: bold;}
    .signal-sell { color: #FF0000; font-weight: bold;}
    .signal-neutral { color: #FFA500; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

from dataclasses import dataclass
from datetime import datetime
from typing import Dict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle
from pytrends.request import TrendReq

@dataclass
class IndicatorSignal:
    name: str
    value: float
    signal: str  # 'buy', 'sell', 'neutral', etc.
    strength: float  # Entre 0 et 1
    description: str

class EnhancedTechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def add_all_indicators(self) -> pd.DataFrame:
        try:
            self.add_moving_averages()
            self.add_macd()
            self.add_adx()
            self.add_rsi()
            self.add_bollinger_bands()
            self.add_volume_indicators()
            return self.data
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Erreur lors du calcul des indicateurs: {e}")
            return self.data

    def add_moving_averages(self):
        if 'close' not in self.data.columns:
            raise ValueError("Colonne 'close' manquante pour les moyennes mobiles.")
        for period in [20, 50, 100, 200]:
            self.data[f'SMA_{period}'] = ta.trend.sma_indicator(self.data['close'], window=period)
            self.data[f'EMA_{period}'] = ta.trend.ema_indicator(self.data['close'], window=period)

    def add_macd(self):
        if 'close' not in self.data.columns:
            raise ValueError("Colonne 'close' manquante pour le MACD.")
        self.data['MACD'] = ta.trend.macd(self.data['close'])
        self.data['MACD_Signal'] = ta.trend.macd_signal(self.data['close'])
        self.data['MACD_Hist'] = ta.trend.macd_diff(self.data['close'])

    def add_rsi(self):
        if 'close' not in self.data.columns:
            raise ValueError("Colonne 'close' manquante pour le RSI.")
        self.data['RSI'] = ta.momentum.rsi(self.data['close'])

    def add_bollinger_bands(self, window: int = 20):
        if 'close' not in self.data.columns:
            raise ValueError("Colonne 'close' manquante pour les bandes de Bollinger.")
        self.data['BB_Upper'] = ta.volatility.bollinger_hband(self.data['close'], window=window)
        self.data['BB_Middle'] = ta.volatility.bollinger_mavg(self.data['close'], window=window)
        self.data['BB_Lower'] = ta.volatility.bollinger_lband(self.data['close'], window=window)

    def add_adx(self, window: int = 14):
        if not all(col in self.data.columns for col in ['high', 'low', 'close']):
            raise ValueError("Colonnes 'high', 'low', 'close' manquantes pour l'ADX.")
        self.data['ADX'] = ta.trend.adx(self.data['high'], self.data['low'], self.data['close'], window=window)

    def add_volume_indicators(self):
        if not all(col in self.data.columns for col in ['close', 'volume']):
            raise ValueError("Colonnes nécessaires manquantes pour les indicateurs de volume.")
        self.data['OBV'] = ta.volume.on_balance_volume(self.data['close'], self.data['volume'])

class BitcoinDataCollector:
    def __init__(self):
        import logging
        self.logger = logging.getLogger(__name__)
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.timeframes = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }

    def fetch_historical_data(self, years: int = 5, timeframe: str = '1h') -> pd.DataFrame:
        try:
            df = pd.read_csv("bitcoin_5y.csv", parse_dates=["Datetime"])
            df.set_index("Datetime", inplace=True)
            print("Shape du DataFrame téléchargé :", df.shape)
            if df.empty:
                print("Aucune donnée téléchargée. Problème de connexion ou de ticker.")
            else:
                df.to_csv("bitcoin_5y.csv", index=False)
                print("Fichier bitcoin_5y.csv sauvegardé avec", len(df), "lignes.")
            return df
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du CSV : {e}")
            return pd.DataFrame()

    def get_market_metrics(self) -> Dict:
        symbol = "BTC-USD"
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2d", interval="1h")
            if data.empty:
                return {}
            last = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else last
            price = last['Close']
            change_24h = ((last['Close'] - prev['Close']) / prev['Close']) * 100 if prev['Close'] != 0 else 0
            volume_24h = data['Volume'][-24:].sum() if len(data) >= 24 else data['Volume'].sum()
            high_24h = data['High'][-24:].max() if len(data) >= 24 else data['High'].max()
            low_24h = data['Low'][-24:].min() if len(data) >= 24 else data['Low'].min()
            return {
                'price': price,
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'bid': price,
                'ask': price,
                'spread': 0,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Erreur métriques marché yfinance: {e}")
            return {}

def get_indicator_signals(df):
    signals = []
    
    # Signal RSI
    if 'RSI' in df.columns and not df['RSI'].dropna().empty:
        rsi = df['RSI'].dropna().iloc[-1]
        if rsi > 70:
            signals.append(IndicatorSignal("RSI", rsi, "sell", abs(rsi-70)/30, "Surachat"))
        elif rsi < 30:
            signals.append(IndicatorSignal("RSI", rsi, "buy", abs(rsi-30)/30, "Survente"))
        else:
            signals.append(IndicatorSignal("RSI", rsi, "neutral", 1-abs(rsi-50)/20, "Zone neutre"))
    
    # Signal MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns and not df['MACD'].dropna().empty and not df['MACD_Signal'].dropna().empty:
        macd = df['MACD'].dropna().iloc[-1]
        macd_signal = df['MACD_Signal'].dropna().iloc[-1]
        if macd > macd_signal:
            signals.append(IndicatorSignal("MACD", macd, "buy", min(abs(macd-macd_signal)/2,1), "Tendance haussière"))
        elif macd < macd_signal:
            signals.append(IndicatorSignal("MACD", macd, "sell", min(abs(macd-macd_signal)/2,1), "Tendance baissière"))
        else:
            signals.append(IndicatorSignal("MACD", macd, "neutral", 0.5, "Croisement neutre"))
    
    # Signal Bandes de Bollinger
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and 'close' in df.columns and not df['BB_Upper'].dropna().empty and not df['BB_Lower'].dropna().empty:
        close = df['close'].dropna().iloc[-1]
        upper = df['BB_Upper'].dropna().iloc[-1]
        lower = df['BB_Lower'].dropna().iloc[-1]
        if close > upper:
            signals.append(IndicatorSignal("Bollinger", close, "sell", min((close-upper)/upper,1), "Cassure bande supérieure"))
        elif close < lower:
            signals.append(IndicatorSignal("Bollinger", close, "buy", min((lower-close)/lower,1), "Cassure bande inférieure"))
        else:
            signals.append(IndicatorSignal("Bollinger", close, "neutral", 0.5, "Dans les bandes"))
    
    # Signal ADX
    if 'ADX' in df.columns and not df['ADX'].dropna().empty:
        adx = df['ADX'].dropna().iloc[-1]
        if adx > 25:
            signals.append(IndicatorSignal("ADX", adx, "trend", min((adx-25)/25,1), "Tendance forte"))
        else:
            signals.append(IndicatorSignal("ADX", adx, "range", 1-(adx/25), "Tendance faible"))
    
    # Signal OBV
    if 'OBV' in df.columns and not df['OBV'].dropna().empty:
        obv = df['OBV'].dropna().iloc[-1]
        signals.append(IndicatorSignal("OBV", obv, "neutral", 0.5, "Volume"))
    
    return signals

def display_market_data(df: pd.DataFrame, metrics: Dict):
    if df.empty:
        st.warning("Aucune donnée de marché disponible.")
        return None
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Prix Bitcoin', 'Volume'),
        row_heights=[0.7, 0.3]
    )
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
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume'
        ),
        row=2, col=1
    )
    fig.update_layout(
        title='Analyse Bitcoin en Temps Réel',
        yaxis_title='Prix (USDT)',
        yaxis2_title='Volume',
        template='plotly_dark',
        height=800,
        xaxis_rangeslider_visible=False
    )
    return fig

def train_ml_model(df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = df[['open', 'high', 'low', 'close', 'volume', 'target']].dropna()
    target = features['target']
    features = features[['open', 'high', 'low', 'close', 'volume']]

    if features.empty or target.empty or len(features) < 10:
        raise ValueError("Pas assez de données pour entraîner le modèle ML.")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

def save_ml_model(model, scaler, model_path="ml_model.pkl", scaler_path="ml_scaler.pkl"):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

def load_ml_model(model_path="ml_model.pkl", scaler_path="ml_scaler.pkl"):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

def predict_ml_signal(model, scaler, last_row: pd.DataFrame):
    last_row_scaled = scaler.transform(last_row[['open', 'high', 'low', 'close', 'volume']])
    prediction = model.predict(last_row_scaled)
    probability = model.predict_proba(last_row_scaled).max()
    return prediction[0], probability

pytrends = TrendReq()
pytrends.build_payload(['bitcoin'], timeframe='today 5-y')
trend_data = pytrends.interest_over_time()
st.line_chart(trend_data['bitcoin'])

def main():
    st.title("📊 Analyse du Marché Bitcoin")
    collector = BitcoinDataCollector()
    st.sidebar.header("Paramètres")
    timeframe = st.sidebar.selectbox(
        "Intervalle de temps",
        ['1m', '5m', '15m', '1h', '4h', '1d'],
        index=3
    )
    
    periode = st.sidebar.selectbox(
        "Période d'analyse",
        ["7 jours", "1 mois", "3 mois", "6 mois", "1 an", "2 ans", "5 ans", "Tout"],
        index=6  # "5 ans" par défaut
    )

    df = collector.fetch_historical_data(years=5, timeframe=timeframe)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
            df = df.set_index('Datetime')
        else:
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    df = df[~df.index.isna()]
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.isnull().any() or len(df) == 0:
        st.error("Impossible de créer un index de dates valide pour le DataFrame.")
        st.stop()

    # Filtrage de l'index pour éliminer les valeurs manquantes
    df = df[df.index.notnull()]

    if periode != "Tout":
        nb_jours = {
            "7 jours": 7,
            "1 mois": 30,
            "3 mois": 90,
            "6 mois": 180,
            "1 an": 365,
            "2 ans": 730,
            "5 ans": 1825
        }[periode]
        last_date = df.index.max()
        date_min = last_date - pd.Timedelta(days=nb_jours)
        df = df[df.index >= date_min]

    st.write("Nombre de lignes après filtrage :", len(df))

    if len(df) < 50:
        st.warning("Pas assez de données pour afficher les indicateurs techniques. Veuillez élargir la période ou l'intervalle.")
        return

    indicators = EnhancedTechnicalIndicators(df)
    df = indicators.add_all_indicators()
    signals = get_indicator_signals(df)

    if 'RSI' in df.columns and not df['RSI'].dropna().empty:
        st.metric("RSI", round(df['RSI'].dropna().iloc[-1], 2))
    else:
        st.warning("RSI non disponible.")

    if 'MACD' in df.columns and not df['MACD'].dropna().empty:
        st.metric("MACD", round(df['MACD'].dropna().iloc[-1], 2))
    else:
        st.warning("MACD non disponible.")

    if 'BB_Upper' in df.columns and not df['BB_Upper'].dropna().empty:
        st.metric("Bollinger Upper", round(df['BB_Upper'].dropna().iloc[-1], 2))
    else:
        st.warning("Bollinger Upper non disponible.")

    if 'ADX' in df.columns and not df['ADX'].dropna().empty:
        st.metric("ADX", round(df['ADX'].dropna().iloc[-1], 2))
    else:
        st.warning("ADX non disponible.")

    if 'OBV' in df.columns and not df['OBV'].dropna().empty:
        st.metric("OBV", round(df['OBV'].dropna().iloc[-1], 2))
    else:
        st.warning("OBV non disponible.")

    st.write("Shape du DataFrame :", df.shape)
    st.write(df.head())
    metrics = collector.get_market_metrics()
    if not df.empty and metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col2:
            st.metric("Spread", f"${metrics['spread']:,.2f}")
        with col4:
            st.metric("Amplitude 24h", f"${metrics['high_24h'] - metrics['low_24h']:,.2f}")
        fig = display_market_data(df, metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Statistiques")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Statistiques de prix")
            st.dataframe(df['close'].describe())
        with col2:
            st.write("Statistiques de volume")
            st.dataframe(df['volume'].describe())
        if len(df) > 30:
            try:
                model, scaler = load_ml_model()
                if model is None or scaler is None:
                    model, scaler = train_ml_model(df)
                    save_ml_model(model, scaler)
                last_row = df.iloc[[-1]]
                ml_pred, ml_proba = predict_ml_signal(model, scaler, last_row)
                ml_label = "Hausse probable" if ml_pred == 1 else "Baisse probable"
                ml_color = "#00FF00" if ml_pred == 1 else "#FF4B4B"
                ml_arrow = "↑" if ml_pred == 1 else "↓"
                st.markdown(
                    f"""
                    <div style='background-color:#181B24;padding:20px;border-radius:10px;margin-bottom:10px;'>
                        <h3 style='color:white;margin:0;'>Prédiction ML</h3>
                        <h1 style='color:{ml_color};margin:0;'>{ml_arrow} {ml_label}</h1>
                        <p style='color:{ml_color};margin:0;'>Confiance : <b>{ml_proba:.2%}</b></p>
                        <span style='color:gray;font-size:12px;'>Dernière mise à jour: {metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    """, unsafe_allow_html=True
                )

                for sig in signals:
                    color = {"buy": "#00FF00", "sell": "#FF0000", "neutral": "#FFA500", "trend": "#00BFFF", "range": "#CCCCCC"}[sig.signal]
                    st.markdown(
                        f"""
                        <div style='background-color:#23263A;padding:10px 20px;border-radius:8px;margin-bottom:8px;'>
                            <b style='color:{color};'>{sig.name.upper()} : {sig.signal.upper()} (valeur: {sig.value:.2f})</b><br>
                            <span style='color:white;'>{sig.description} | Force: <b>{sig.strength:.2f}</b></span>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    if sig.name == "MACD" and sig.signal == "sell":
                        st.markdown(
                            "<div style='background-color:#FF4B4B;color:white;padding:10px;border-radius:8px;margin-bottom:10px;'>"
                            "<b>ALERTE VENTE (MACD) :</b> Tendance baissière</div>",
                            unsafe_allow_html=True
                        )

                weights = {"RSI": 0.3, "MACD": 0.3, "Bollinger": 0.2, "ADX": 0.2}
                score = 0
                total_weight = 0
                for sig in signals:
                    w = weights.get(sig.name, 0.1)
                    if sig.signal == "buy":
                        score += w * sig.strength
                    elif sig.signal == "sell":
                        score -= w * sig.strength
                    total_weight += w
                if total_weight > 0:
                    score_global = score / total_weight
                    st.markdown(
                        f"<div style='background-color:#23263A;color:white;padding:10px;border-radius:8px;margin-bottom:10px;'>"
                        f"<b>Score Global Synthétique</b> : <span style='color:{'#00FF00' if score_global > 0 else '#FF4B4B'};'>{score_global:.2f}</span> "
                        "<span style='font-size:12px;'>(>0 = biais haussier, <0 = biais baissier)</span></div>",
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.info(f"ML non disponible : {e}")
        else:
            st.info("Pas assez de données pour entraîner le modèle ML.")
        st.caption(f"Dernière mise à jour: {metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("Aucune donnée disponible pour l'affichage.")

    weights = {"RSI": 0.3, "MACD": 0.3, "Bollinger": 0.2, "ADX": 0.2}
    score = 0
    total_weight = 0
    for sig in signals:
        w = weights.get(sig.name, 0.1)
        if sig.signal == "buy":
            score += w * sig.strength
        elif sig.signal == "sell":
            score -= w * sig.strength
        total_weight += w

    if total_weight > 0:
        score_global = score / total_weight
        st.metric("Score Global Synthétique", f"{score_global:.2f}", help=">0 = biais haussier, <0 = biais baissier")
    
    if st.button("Optimiser le modèle ML"):
        if len(df) > 30:
            model, scaler = train_ml_model(df)
            save_ml_model(model, scaler)
            st.success("Modèle ML optimisé et sauvegardé !")
        else:
            st.warning("Pas assez de données pour optimiser le modèle ML.")

if __name__ == "__main__":
    main()