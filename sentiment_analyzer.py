import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import logging
from dataclasses import dataclass
import yfinance as yf
import pickle
from pytrends.request import TrendReq
import requests

# --------- CLASSES ---------
@dataclass
class IndicatorSignal:
    """Classe pour stocker les signaux des indicateurs"""
    name: str
    value: float
    signal: str  # 'buy', 'sell', 'neutral', 'trend', 'range'
    strength: float  # 0 à 1
    description: str

class EnhancedTechnicalIndicators:
    """Ajoute des indicateurs techniques à un DataFrame de marché"""
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
    """Collecteur de données Bitcoin via yfinance"""
    def __init__(self):
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
        """Charge les données historiques depuis le CSV local."""
        try:
            df = pd.read_csv("bitcoin_5y.csv", parse_dates=["Datetime"])
            df.set_index("Datetime", inplace=True)
            for col in ['close', 'high', 'low', 'open', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du CSV : {e}")
            return pd.DataFrame()

    def get_market_metrics(self) -> dict:
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

# --------- INDICATEURS ET SIGNAUX ---------
def get_indicator_signals(df):
    signals = []
    # RSI
    if 'RSI' in df.columns and not df['RSI'].isna().all():
        rsi = df['RSI'].dropna().iloc[-1]
        if rsi > 70:
            signals.append(IndicatorSignal("RSI", rsi, "sell", abs(rsi-70)/30, "Surachat"))
        elif rsi < 30:
            signals.append(IndicatorSignal("RSI", rsi, "buy", abs(rsi-30)/30, "Survente"))
        else:
            signals.append(IndicatorSignal("RSI", rsi, "neutral", 1-abs(rsi-50)/20, "Zone neutre"))
    # MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = df['MACD'].dropna().iloc[-1]
        macd_signal = df['MACD_Signal'].dropna().iloc[-1]
        if macd > macd_signal:
            signals.append(IndicatorSignal("MACD", macd, "buy", min(abs(macd-macd_signal)/2,1), "Tendance haussière"))
        elif macd < macd_signal:
            signals.append(IndicatorSignal("MACD", macd, "sell", min(abs(macd-macd_signal)/2,1), "Tendance baissière"))
        else:
            signals.append(IndicatorSignal("MACD", macd, "neutral", 0.5, "Croisement neutre"))
    # Bandes de Bollinger
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns and 'close' in df.columns:
        close = df['close'].dropna().iloc[-1]
        upper = df['BB_Upper'].dropna().iloc[-1]
        lower = df['BB_Lower'].dropna().iloc[-1]
        if close > upper:
            signals.append(IndicatorSignal("Bollinger", close, "sell", min((close-upper)/upper,1), "Cassure bande supérieure"))
        elif close < lower:
            signals.append(IndicatorSignal("Bollinger", close, "buy", min((lower-close)/lower,1), "Cassure bande inférieure"))
        else:
            signals.append(IndicatorSignal("Bollinger", close, "neutral", 0.5, "Dans les bandes"))
    # ADX
    if 'ADX' in df.columns:
        adx = df['ADX'].dropna().iloc[-1]
        if adx > 25:
            signals.append(IndicatorSignal("ADX", adx, "trend", min((adx-25)/25,1), "Tendance forte"))
        else:
            signals.append(IndicatorSignal("ADX", adx, "range", 1-(adx/25), "Tendance faible"))
    # OBV
    if 'OBV' in df.columns:
        obv = df['OBV'].dropna().iloc[-1]
        signals.append(IndicatorSignal("OBV", obv, "neutral", 0.5, "Volume"))
    return signals

# --------- ML ---------
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

# --------- DASHBOARD ---------
def display_market_data(df: pd.DataFrame, metrics: dict):
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

# --------- GOOGLE TRENDS ---------
def display_google_trends():
    pytrends = TrendReq()
    pytrends.build_payload(['bitcoin'], timeframe='today 5-y')
    trend_data = pytrends.interest_over_time()
    st.subheader("Tendance Google Trends (Bitcoin)")
    st.line_chart(trend_data['bitcoin'])

# --------- FEAR & GREED INDEX ---------
def get_fear_greed_index():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        value = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        timestamp = datetime.fromtimestamp(int(data['data'][0]['timestamp']))
        return {"value": value, "classification": classification, "timestamp": timestamp}
    except Exception as e:
        st.error(f"Erreur Fear & Greed Index: {e}")
        return None

# --------- MAIN ---------
def main():
    st.set_page_config(
        page_title="Bitcoin Analytics Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
        .main { background-color: #0E1117; color: #FFFFFF; }
        </style>
    """, unsafe_allow_html=True)
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
        ["7 jours", "1 mois", "3 mois", "6 mois", "1 an", "2 ans", "5 ans", "Tout"]
    )
    df = collector.fetch_historical_data(years=5, timeframe=timeframe)
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
        date_min = df.index.max() - pd.Timedelta(days=nb_jours)
        df = df[df.index >= date_min]
    st.write("Nombre de lignes après filtrage :", len(df))
    indicators = EnhancedTechnicalIndicators(df)
    df = indicators.add_all_indicators()
    if 'RSI' in df.columns and not df['RSI'].isna().all():
        st.metric("RSI", round(df['RSI'].dropna().iloc[-1], 2))
    else:
        st.warning("RSI non disponible.")
    if 'MACD' in df.columns:
        st.metric("MACD", round(df['MACD'].dropna().iloc[-1], 2))
    st.write("Shape du DataFrame :", df.shape)
    st.write(df.head())
    metrics = collector.get_market_metrics()
    if not df.empty and metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prix Bitcoin", f"${metrics['price']:,.2f}", f"{metrics['change_24h']:+.2f}%")
        with col2:
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
                st.metric("Prédiction ML", ml_label, f"Confiance : {ml_proba:.2%}")
            except Exception as e:
                st.info(f"ML non disponible : {e}")
        else:
            st.info("Pas assez de données pour entraîner le modèle ML.")
        st.caption(f"Dernière mise à jour: {metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("Aucune donnée disponible pour l'affichage.")

    # Affichage des moyennes mobiles calculables
    st.subheader("Moyennes mobiles disponibles")
    ma_cols = [col for col in df.columns if col.startswith("SMA_") or col.startswith("EMA_")]
    ma_to_show = {}
    for col in ma_cols:
        last_val = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
        if last_val is not None:
            ma_to_show[col] = last_val

    if ma_to_show:
        for name, value in ma_to_show.items():
            st.metric(name, f"{value:.2f}")
    else:
        st.info("Aucune moyenne mobile calculable sur la période affichée.")

    # Signaux et alertes
    signals = get_indicator_signals(df)
    for sig in signals:
        color = {"buy": "#00FF00", "sell": "#FF0000", "neutral": "#FFA500", "trend": "#00BFFF", "range": "#CCCCCC"}[sig.signal]
        st.markdown(
            f"<div class='indicator-box' style='color:{color};'>"
            f"<b>{sig.name}</b> : {sig.signal.upper()} (valeur: {sig.value:.2f})<br>"
            f"<small>{sig.description} | Force: {sig.strength:.2f}</small>"
            "</div>", unsafe_allow_html=True
        )
        if sig.signal == "buy" and sig.strength > 0.7:
            st.success(f"ALERTE ACHAT ({sig.name}) : {sig.description}")
        if sig.signal == "sell" and sig.strength > 0.7:
            st.error(f"ALERTE VENTE ({sig.name}) : {sig.description}")

    # Score global pondéré
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

    # Google Trends
    display_google_trends()

    # Fear & Greed Index
    fg = get_fear_greed_index()
    if fg:
        st.metric("Fear & Greed Index", fg["value"], fg["classification"])
        st.caption(f"Dernière mise à jour: {fg['timestamp']}")

    # Optimisation du modèle ML
    if st.button("Optimiser le modèle ML"):
        model, scaler = train_ml_model(df)
        save_ml_model(model, scaler)
        st.success("Modèle ML optimisé et sauvegardé !")

if __name__ == "__main__":
    main()