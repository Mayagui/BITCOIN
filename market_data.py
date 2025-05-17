import os
import pandas as pd
import streamlit as st
import ta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(
    page_title="Bitcoin Analytics",
    page_icon="₿",
    layout="wide"
)

# --- CSS custom ---
def custom_css():
    st.markdown("""
    <style>
    .main {background-color: #181818;}
    .stMetric {font-size: 1.2em;}
    </style>
    """, unsafe_allow_html=True)

# --- Utilitaires ---
def rename_ohlcv_columns(df):
    mapping = {
        "Date": "timestamp",
        "Datetime": "timestamp",
        "Close": "close",
        "High": "high",
        "Low": "low",
        "Open": "open",
        "Volume": "volume"
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

@st.cache_data(show_spinner="Chargement des données...")
def load_bitcoin_data():
    if not os.path.exists("bitcoin_5y.csv"):
        df = yf.download("BTC-USD", period="5y", interval="1d")
        df.reset_index(inplace=True)
        df = rename_ohlcv_columns(df)
        if "timestamp" not in df.columns or df["timestamp"].isnull().all():
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.to_csv("bitcoin_5y.csv", index=False)
    csv_path = 'bitcoin_5y.csv'
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns and "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = rename_ohlcv_columns(df)
    df = df.dropna(subset=['timestamp'])
    df = df.set_index('timestamp')
    for col in ['close', 'high', 'low', 'open', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data(show_spinner="Calcul des indicateurs...")
def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(window=30).std() * (365 ** 0.5)
    df["SMA_50"] = df["close"].rolling(window=50).mean()
    df["SMA_200"] = df["close"].rolling(window=200).mean()
    cross = (df["SMA_50"] > df["SMA_200"]) & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
    df.loc[cross, "signal"] = "golden_cross"
    df["trend"] = df["close"].diff().apply(lambda x: "up" if x > 0 else "down")
    df["MACD"] = ta.trend.macd(df["close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["close"])
    df["MACD_Hist"] = ta.trend.macd_diff(df["close"])
    df["RSI"] = ta.momentum.rsi(df["close"])
    df["Stoch_RSI"] = ta.momentum.stochrsi(df["close"])
    df["Williams_%R"] = ta.momentum.williams_r(df["high"], df["low"], df["close"])
    df["CCI"] = ta.trend.cci(df["high"], df["low"], df["close"])
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    df["BB_Upper"] = ta.volatility.bollinger_hband(df["close"])
    df["BB_Middle"] = ta.volatility.bollinger_mavg(df["close"])
    df["BB_Lower"] = ta.volatility.bollinger_lband(df["close"])
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"])
    df["OBV"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    return df

@st.cache_data(show_spinner="Chargement Google Trends...")
def load_google_trends():
    # Exemple simple, à remplacer par pytrends si besoin
    dates = pd.date_range(end=pd.Timestamp.today(), periods=365)
    trends = pd.DataFrame({
        "date": dates,
        "bitcoin": np.random.randint(20, 100, size=len(dates)),
        "crypto": np.random.randint(10, 80, size=len(dates)),
        "BTC": np.random.randint(5, 60, size=len(dates))
    }).set_index("date")
    return trends

def compute_signals(df):
    df = df.copy()
    df["signal_rsi"] = np.where(df["RSI"] < 30, "Achat", np.where(df["RSI"] > 70, "Vente", "Neutre"))
    df["signal_macd"] = np.where(df["MACD"] > df["MACD_Signal"], "Achat", "Vente")
    df["signal_stochrsi"] = np.where(df["Stoch_RSI"] < 0.2, "Achat", np.where(df["Stoch_RSI"] > 0.8, "Vente", "Neutre"))
    df["signal_williams"] = np.where(df["Williams_%R"] < -80, "Achat", np.where(df["Williams_%R"] > -20, "Vente", "Neutre"))
    df["score"] = (
        (df["signal_rsi"] == "Achat").astype(int) +
        (df["signal_macd"] == "Achat").astype(int) +
        (df["signal_stochrsi"] == "Achat").astype(int) +
        (df["signal_williams"] == "Achat").astype(int)
    )
    return df

@st.cache_data(show_spinner="Calcul ML...")
def compute_ml(df):
    """Exemple ML RandomForest sur la tendance."""
    features = df[["open", "high", "low", "close", "volume"]].dropna()
    target = (df["close"].shift(-1) > df["close"]).astype(int).dropna()
    features = features.iloc[:-1]
    target = target.iloc[:len(features)]
    if len(features) < 30:
        return None, None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, target)
    last_row = features.iloc[[-1]]
    pred = model.predict(scaler.transform(last_row))[0]
    proba = model.predict_proba(scaler.transform(last_row))[0][1]
    return pred, proba, model

# --- Application principale ---

def main():
    custom_css()
    st.title("📊 Analyse du Marché Bitcoin (Optimisée & Complète)")

    onglet = st.sidebar.radio("Navigation", [
        "Vue Marché", "Indicateurs", "Signaux", "ML", "Google Trends", "Données brutes"
    ])

    df = load_bitcoin_data()
    df = compute_indicators(df)
    trends = load_google_trends()

    # Sélection de la période
    st.sidebar.header("Période d'analyse")
    periode = st.sidebar.selectbox(
        "Période",
        ["7 jours", "1 mois", "3 mois", "6 mois", "1 an", "2 ans", "5 ans", "Tout"],
        index=6
    )
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

    if onglet == "Vue Marché":
        st.metric("RSI", round(df['RSI'].dropna().iloc[-1], 2) if 'RSI' in df.columns and not df['RSI'].dropna().empty else "N/A")
        st.metric("MACD", round(df['MACD'].dropna().iloc[-1], 2) if 'MACD' in df.columns and not df['MACD'].dropna().empty else "N/A")
        st.metric("Bollinger Upper", round(df['BB_Upper'].dropna().iloc[-1], 2) if 'BB_Upper' in df.columns and not df['BB_Upper'].dropna().empty else "N/A")
        st.metric("ADX", round(df['ADX'].dropna().iloc[-1], 2) if 'ADX' in df.columns and not df['ADX'].dropna().empty else "N/A")
        st.metric("OBV", round(df['OBV'].dropna().iloc[-1], 2) if 'OBV' in df.columns and not df['OBV'].dropna().empty else "N/A")
        st.write("Shape du DataFrame :", df.shape)
        st.write(df.head())

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
        st.plotly_chart(fig, use_container_width=True)

    elif onglet == "Indicateurs":
        st.subheader("Indicateurs techniques")
        st.dataframe(df.tail(30))

        with st.expander("📈 RSI"):
            st.line_chart(df["RSI"].dropna())

        with st.expander("📈 MACD & Signal"):
            st.line_chart(df[["MACD", "MACD_Signal"]].dropna())

        with st.expander("📈 Bandes de Bollinger"):
            st.line_chart(df[["close", "BB_Upper", "BB_Lower"]].dropna())

        with st.expander("📈 ADX"):
            st.line_chart(df["ADX"].dropna())

        with st.expander("📈 OBV"):
            st.line_chart(df["OBV"].dropna())

    elif onglet == "Signaux":
        st.subheader("Signaux personnalisés & scoring")
        df_signals = compute_signals(df)
        st.dataframe(df_signals[["RSI", "MACD", "signal_rsi", "signal_macd", "signal_stochrsi", "signal_williams", "score"]].tail(30))

    elif onglet == "ML":
        st.subheader("Machine Learning (RandomForest)")

        periode_ml = st.selectbox(
            "Période d'entraînement ML",
            ["30 jours", "90 jours", "180 jours", "1 an", "2 ans", "5 ans", "Tout"],
            index=3
        )
        nb_jours_ml = {
            "30 jours": 30,
            "90 jours": 90,
            "180 jours": 180,
            "1 an": 365,
            "2 ans": 730,
            "5 ans": 1825,
            "Tout": len(df)
        }[periode_ml]
        df_ml = df.tail(nb_jours_ml)

        modele = st.selectbox("Modèle ML", ["RandomForest", "LogisticRegression"], index=0)

        if modele == "RandomForest":
            pred, proba, model = compute_ml(df_ml)
        else:
            from sklearn.linear_model import LogisticRegression
            features = df_ml[["open", "high", "low", "close", "volume"]].dropna()
            target = (df_ml["close"].shift(-1) > df_ml["close"]).astype(int).dropna()
            features = features.iloc[:-1]
            target = target.iloc[:len(features)]
            if len(features) < 30:
                pred, proba, model = None, None, None
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(features)
                model = LogisticRegression()
                model.fit(X_scaled, target)
                last_row = features.iloc[[-1]]
                pred = model.predict(scaler.transform(last_row))[0]
                proba = model.predict_proba(scaler.transform(last_row))[0][1]

        if model is not None:
            st.metric("Prédiction ML", "Hausse probable" if pred == 1 else "Baisse probable", f"Confiance : {proba:.2%}")
        else:
            st.info("Pas assez de données pour entraîner le modèle ML.")

    elif onglet == "Google Trends":
        st.subheader("Google Trends (exemple)")
        st.line_chart(trends)

    elif onglet == "Données brutes":
        st.dataframe(df)

if __name__ == "__main__":
    main()