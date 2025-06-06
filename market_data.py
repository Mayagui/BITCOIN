import os
import pandas as pd
import streamlit as st
import ta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import io
from data_collector import BitcoinDataCollector as DataCollector
import requests

# Configuration de la page
st.set_page_config(
    page_title="Bitcoin Analytics",
    page_icon="₿",
    layout="wide"
)

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

    /* Thème clair */
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

    # Ajout du calcul des rendements
    df['returns'] = df['close'].pct_change()

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
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    df["BB_Upper"] = ta.volatility.bollinger_hband(df["close"])
    df["BB_Middle"] = ta.volatility.bollinger_mavg(df["close"])
    df["BB_Lower"] = ta.volatility.bollinger_lband(df["close"])
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"])
    df["OBV"] = ta.volume.on_balance_volume(df["close"], df["volume"])
    return df

def compute_moving_average_crossover(df):
    # Calcul des moyennes mobiles
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
            
            # Signaux de croisement
    df['Signal_20_50'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
    df['Signal_50_200'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)

    # Points de croisement
    df['Cross_20_50'] = df['Signal_20_50'].diff()
    df['Cross_50_200'] = df['Signal_50_200'].diff()

    return df

def compute_volatility_strategy(df):
    # Calcul de l'ATR
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])

    # Calcul des bandes de Bollinger
    df['BB_Upper'] = ta.volatility.bollinger_hband(df['close'])
    df['BB_Lower'] = ta.volatility.bollinger_lband(df['close'])

    # Signaux de volatilité
    df['Volatility_Signal'] = np.where(
        (df['close'] < df['BB_Lower']) & (df['RSI'] < 30), 1,
        np.where((df['close'] > df['BB_Upper']) & (df['RSI'] > 70), -1, 0)
    )

    return df

def compute_volume_strategy(df):
    # Calcul de l'OBV
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])

    # Calcul du Money Flow Index
    df['MFI'] = ta.volume.money_flow_index(
        df['high'], df['low'], df['close'], df['volume']
        )
        
        # Signaux de volume
    df['Volume_Signal'] = np.where(
        (df['OBV'].diff() > 0) & (df['MFI'] < 20), 1,
        np.where((df['OBV'].diff() < 0) & (df['MFI'] > 80), -1, 0)
    )

    return df

def get_fear_greed_index():
    # Utiliser l'API Fear & Greed Index
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    data = response.json()
    return int(data['data'][0]['value'])

def backtest_strategy(df, initial_capital=100000, position_size=0.1):
    # Calcul des signaux combinés
    df = compute_moving_average_crossover(df)
    df = compute_volatility_strategy(df)
    df = compute_volume_strategy(df)

    # Combinaison des signaux
    df['Combined_Signal'] = (
        df['Signal_20_50'] + 
        df['Signal_50_200'] + 
        df['Volatility_Signal'] + 
        df['Volume_Signal']
    )

    # Simulation de trading
    portfolio = pd.DataFrame(index=df.index)
    portfolio['Position'] = 0
    portfolio['Capital'] = initial_capital
    portfolio['BTC'] = 0

    for i in range(1, len(df)):
        if df['Combined_Signal'].iloc[i] > 2:  # Signal d'achat fort
            if portfolio['Position'].iloc[i-1] == 0:
                btc_to_buy = (portfolio['Capital'].iloc[i-1] * position_size) / df['close'].iloc[i]
                portfolio.loc[df.index[i], 'BTC'] = btc_to_buy
                portfolio.loc[df.index[i], 'Position'] = 1
                portfolio.loc[df.index[i], 'Capital'] = portfolio['Capital'].iloc[i-1] - (btc_to_buy * df['close'].iloc[i])
        elif df['Combined_Signal'].iloc[i] < -2:  # Signal de vente fort
            if portfolio['Position'].iloc[i-1] == 1:
                portfolio.loc[df.index[i], 'Capital'] = portfolio['Capital'].iloc[i-1] + (portfolio['BTC'].iloc[i-1] * df['close'].iloc[i])
                portfolio.loc[df.index[i], 'BTC'] = 0
                portfolio.loc[df.index[i], 'Position'] = 0
        else:
            portfolio.loc[df.index[i], 'Capital'] = portfolio['Capital'].iloc[i-1]
            portfolio.loc[df.index[i], 'BTC'] = portfolio['BTC'].iloc[i-1]
            portfolio.loc[df.index[i], 'Position'] = portfolio['Position'].iloc[i-1]

    return portfolio

def main():
    custom_css()

    # Sélecteur de thème
    theme = st.sidebar.radio(
        "Thème",
        ["🌙 Sombre", "☀️ Clair"],
        index=0
    )

    # Appliquer le thème
    if theme == "☀️ Clair":
        st.markdown("""
        <script>
            document.documentElement.setAttribute('data-theme', 'light');
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <script>
            document.documentElement.setAttribute('data-theme', 'dark');
        </script>
        """, unsafe_allow_html=True)

    st.title("📊 Analyse du Marché Bitcoin (Optimisée & Complète)")

    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="logo.svg" width="120" alt="Bitcoin Logo" />
        </div>
        """,
        unsafe_allow_html=True
    )

    # Chargement des données
    df = load_bitcoin_data()

    # Calcul des indicateurs techniques IMMÉDIATEMENT après le chargement
    df = compute_indicators(df)

    # Sélection de la période
    if "periode" not in st.session_state:
        st.session_state.periode = "5 ans"

    periode = st.sidebar.selectbox(
        "Période",
        ["7 jours", "1 mois", "3 mois", "6 mois", "1 an", "2 ans", "5 ans", "Tout"],
        index=["7 jours", "1 mois", "3 mois", "6 mois", "1 an", "2 ans", "5 ans", "Tout"].index(st.session_state.periode)
    )
    st.session_state.periode = periode

    nb_jours = {
        "7 jours": 7,
        "1 mois": 30,
        "3 mois": 90,
        "6 mois": 180,
        "1 an": 365,
        "2 ans": 730,
        "5 ans": 1825,
        "Tout": None
    }[periode]

    if nb_jours is not None:
        last_date = df.index.max()
        date_min = last_date - pd.Timedelta(days=nb_jours)
        df = df[df.index >= date_min]

    # Navigation
    onglet = st.sidebar.radio(
        "Navigation",
        ["Market Overview", "Technical Analysis", "RSI Trading", "Price Prediction", "Backtest", "Advanced Strategies"]
    )

    if onglet == "Market Overview":
        st.subheader("📈 Vue d'ensemble du marché")

        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = df['close'].iloc[-1]
            st.metric("Prix actuel", f"${current_price:,.2f}")
        with col2:
            daily_change = df['close'].pct_change().iloc[-1] * 100
            st.metric("Variation journalière", f"{daily_change:+.2f}%", 
                     delta_color="normal" if daily_change >= 0 else "inverse")
        with col3:
            weekly_change = df['close'].pct_change(7).iloc[-1] * 100
            st.metric("Variation hebdomadaire", f"{weekly_change:+.2f}%",
                     delta_color="normal" if weekly_change >= 0 else "inverse")
        with col4:
            volatility = df['returns'].rolling(window=30).std().iloc[-1] * 100
            st.metric("Volatilité (30j)", f"{volatility:.2f}%")

        # Graphique des prix avec indicateurs
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC/USD'
        ), row=1, col=1)

        # Moyennes mobiles
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='orange')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_200'],
            name='SMA 200',
            line=dict(color='blue')
        ), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color='rgba(0, 255, 0, 0.3)'
        ), row=2, col=1)

        fig.update_layout(
            title='Prix du Bitcoin et Volume',
            yaxis_title='Prix (USD)',
            yaxis2_title='Volume',
            xaxis_title='Date',
            template='plotly_dark',
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistiques supplémentaires
        st.subheader("📊 Statistiques du marché")
        col1, col2 = st.columns(2)

        with col1:
            # Distribution des rendements
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Histogram(
                x=df['returns'] * 100,
                nbinsx=50,
                name='Distribution des rendements'
            ))
            fig_returns.update_layout(
                title='Distribution des rendements journaliers',
                xaxis_title='Rendement (%)',
                yaxis_title='Fréquence',
                template='plotly_dark'
            )
            st.plotly_chart(fig_returns, use_container_width=True)

        with col2:
            # Corrélation volume-prix
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df['volume'],
                y=df['close'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['returns'] * 100,
                    colorscale='RdYlGn',
                    showscale=True
                ),
                name='Volume vs Prix'
            ))
            fig_corr.update_layout(
                title='Corrélation Volume-Prix',
                xaxis_title='Volume',
                yaxis_title='Prix (USD)',
                template='plotly_dark'
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    elif onglet == "Technical Analysis":
        st.subheader("📊 Analyse Technique")

        # Options d'analyse
        st.sidebar.subheader("Options d'analyse")
        show_rsi = st.sidebar.checkbox("Afficher RSI", value=True)
        show_macd = st.sidebar.checkbox("Afficher MACD", value=True)
        show_bb = st.sidebar.checkbox("Afficher Bollinger Bands", value=True)
        show_adx = st.sidebar.checkbox("Afficher ADX", value=True)

        # Graphique principal avec sous-graphiques
        fig = make_subplots(rows=4, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.4, 0.2, 0.2, 0.2])

        # Prix et Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC/USD'
        ), row=1, col=1)

        if show_bb:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Sup',
                line=dict(color='rgba(255,0,0,0.5)')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Inf',
                line=dict(color='rgba(0,255,0,0.5)')
            ), row=1, col=1)

        # RSI
        if show_rsi:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI'
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        if show_macd:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD'
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal'
            ), row=3, col=1)
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['MACD_Hist'],
                name='Histogram'
            ), row=3, col=1)

        # ADX
        if show_adx:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['ADX'],
                name='ADX'
            ), row=4, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="yellow", row=4, col=1)

        fig.update_layout(
            title='Analyse Technique Complète',
            height=1000,
            template='plotly_dark',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Signaux techniques
        st.subheader("Signaux Techniques")
        signals = pd.DataFrame(index=df.index)
        signals['RSI'] = np.where(df['RSI'] < 30, 'Achat', np.where(df['RSI'] > 70, 'Vente', 'Neutre'))
        signals['MACD'] = np.where(df['MACD'] > df['MACD_Signal'], 'Haussier', 'Baissier')
        signals['BB'] = np.where(df['close'] < df['BB_Lower'], 'Achat', 
                               np.where(df['close'] > df['BB_Upper'], 'Vente', 'Neutre'))
        signals['ADX'] = np.where(df['ADX'] > 25, 'Tendance Forte', 'Tendance Faible')

        st.dataframe(signals.tail(10))

    elif onglet == "RSI Trading":
        st.subheader("📊 Stratégie de Trading basée sur le RSI")

        # Paramètres de la stratégie RSI
        st.sidebar.subheader("Paramètres de la stratégie RSI")
        rsi_period = st.sidebar.slider("Période RSI", 5, 30, 14)
        rsi_oversold = st.sidebar.slider("Niveau de survente", 20, 40, 30)
        rsi_overbought = st.sidebar.slider("Niveau de surachat", 60, 80, 70)

        # Filtres supplémentaires
        st.sidebar.subheader("Filtres de confirmation")
        use_macd = st.sidebar.checkbox("Utiliser MACD comme confirmation", value=True)
        use_volume = st.sidebar.checkbox("Utiliser le volume comme confirmation", value=True)
        min_volume_multiplier = st.sidebar.slider("Multiplicateur de volume minimum", 1.0, 3.0, 1.5)

        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_rsi = df['RSI'].iloc[-1]
            st.metric("RSI actuel", f"{current_rsi:.2f}", 
                     delta="Survente" if current_rsi < rsi_oversold else "Surachat" if current_rsi > rsi_overbought else "Neutre")
        with col2:
            st.metric("Signal MACD", "Haussier" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "Baissier")
        with col3:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            st.metric("Ratio de volume", f"{volume_ratio:.2f}x")
        with col4:
            st.metric("Tendance", "Haussière" if df['close'].iloc[-1] > df['SMA_50'].iloc[-1] else "Baissière")

        # Graphique principal avec RSI et signaux
        fig = make_subplots(rows=3, cols=1, 
        shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.5, 0.25, 0.25])

        # Prix
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC/USD'
        ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI'
        ), row=2, col=1)

        # Zones RSI
        fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color='rgba(0, 255, 0, 0.3)'
        ), row=3, col=1)

        # Ajout des signaux d'achat/vente
        buy_signals = (df['RSI'] < rsi_oversold) & (df['MACD'] > df['MACD_Signal']) & (df['volume'] > df['volume'].rolling(20).mean() * min_volume_multiplier)
        sell_signals = (df['RSI'] > rsi_overbought) & (df['MACD'] < df['MACD_Signal']) & (df['volume'] > df['volume'].rolling(20).mean() * min_volume_multiplier)

        fig.add_trace(go.Scatter(
            x=df.index[buy_signals],
            y=df['close'][buy_signals],
            mode='markers',
            name='Signal Achat',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index[sell_signals],
            y=df['close'][sell_signals],
            mode='markers',
            name='Signal Vente',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ), row=1, col=1)

    fig.update_layout(
            title='Signaux de Trading RSI',
        height=800,
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Analyse des signaux
        st.subheader("Analyse des Signaux")

        # Derniers signaux
        last_signals = pd.DataFrame({
            'Date': df.index[-10:],
            'RSI': df['RSI'].iloc[-10:],
            'MACD': df['MACD'].iloc[-10:],
            'Volume Ratio': df['volume'].iloc[-10:] / df['volume'].rolling(20).mean().iloc[-10:],
            'Signal': np.where(buy_signals.iloc[-10:], 'Achat', 
                             np.where(sell_signals.iloc[-10:], 'Vente', 'Neutre'))
        })
        st.dataframe(last_signals)

        # Recommandation actuelle
        st.subheader("Recommandation de Trading")
        current_signal = "Achat" if buy_signals.iloc[-1] else "Vente" if sell_signals.iloc[-1] else "Neutre"

        if current_signal == "Achat":
            st.success(f"Signal d'achat détecté! RSI: {current_rsi:.2f}, MACD: Haussier, Volume: {volume_ratio:.2f}x")
        elif current_signal == "Vente":
            st.warning(f"Signal de vente détecté! RSI: {current_rsi:.2f}, MACD: Baissier, Volume: {volume_ratio:.2f}x")
        else:
            st.info(f"Pas de signal clair. RSI: {current_rsi:.2f}, Attendez une confirmation.")

        # Statistiques des signaux
        st.subheader("Statistiques des Signaux")
        col1, col2 = st.columns(2)

        with col1:
            # Distribution des signaux
            signal_dist = pd.Series(np.where(buy_signals, 'Achat', 
                                           np.where(sell_signals, 'Vente', 'Neutre'))).value_counts()
            fig_dist = go.Figure(data=[go.Pie(
                labels=signal_dist.index,
                values=signal_dist.values,
                hole=.3
            )])
            fig_dist.update_layout(title='Distribution des Signaux')
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # Performance des signaux
            signal_returns = pd.DataFrame({
                'Signal': np.where(buy_signals, 'Achat', 
                                 np.where(sell_signals, 'Vente', 'Neutre')),
                'Return': df['returns'] * 100
            })
            signal_performance = signal_returns.groupby('Signal')['Return'].agg(['mean', 'std', 'count'])
            st.dataframe(signal_performance)

    elif onglet == "Price Prediction":
        st.subheader("🔮 Prédiction de Prix")

        # Préparation des features
        features = df[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_Signal', 
                      'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR', 'ADX', 'CCI', 'Williams_%R']].copy()

        # Nettoyage des données
        features = features.fillna(method='ffill').fillna(method='bfill')

        # Target : prix futur (1 jour plus tard)
        target = df['close'].shift(-1)

        # Supprimer les lignes avec des valeurs NaN
        valid_idx = ~target.isna()
        features = features[valid_idx]
        target = target[valid_idx]

        # Vérification que les données sont alignées
        if len(features) != len(target):
            st.error("Erreur dans la préparation des données. Veuillez réessayer.")
            return

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Entraînement des modèles
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        predictions = {}
        metrics = {}

        for name, model in models.items():
            # Entraînement
            model.fit(X_train, y_train)

            # Prédiction
            last_features = features.iloc[[-1]]
            predictions[name] = model.predict(last_features)[0]

            # Métriques
            y_pred = model.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = model.score(X_test, y_test)

            metrics[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }

        # Affichage des résultats
        st.subheader("Prédictions pour le prochain jour")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prix actuel", f"${df['close'].iloc[-1]:,.2f}")
        with col2:
            st.metric("Meilleure prédiction", f"${max(predictions.values()):,.2f}")

        # Graphique des prédictions vs réalité
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-30:],
            y=df['close'].iloc[-30:],
            name='Prix réel',
            line=dict(color='blue')
        ))

        for name, pred in predictions.items():
            fig.add_trace(go.Scatter(
                x=[df.index[-1]],
                y=[pred],
                mode='markers',
                name=f'Prédiction {name}',
                marker=dict(size=10)
            ))

        fig.update_layout(
            title='Prédictions de prix vs réalité',
            yaxis_title='Prix (USD)',
            xaxis_title='Date',
            template='plotly_dark'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Affichage des métriques de performance
        st.subheader("Métriques de Performance par Modèle")
        for name, metric in metrics.items():
            st.write(f"**{name}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"${metric['RMSE']:,.2f}")
            with col2:
                st.metric("MAE", f"${metric['MAE']:,.2f}")
            with col3:
                st.metric("R²", f"{metric['R²']:.2f}")

    elif onglet == "Backtest":
        st.subheader("📊 Backtest de Stratégie")

        # Paramètres de la stratégie
        st.sidebar.subheader("Paramètres de la stratégie")

        # Paramètres RSI
        st.sidebar.write("**Paramètres RSI**")
        rsi_period = st.sidebar.slider("Période RSI", 5, 30, 14)
        rsi_oversold = st.sidebar.slider("Niveau de survente RSI", 20, 40, 30)
        rsi_overbought = st.sidebar.slider("Niveau de surachat RSI", 60, 80, 70)

        # Paramètres de gestion du risque
        st.sidebar.write("**Gestion du risque**")
        stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 10, 5)
        take_profit = st.sidebar.slider("Take Profit (%)", 5, 30, 15)

        # Paramètres de volume
        st.sidebar.write("**Paramètres de volume**")
        volume_ma_period = st.sidebar.slider("Période moyenne mobile volume", 10, 50, 20)
        volume_threshold = st.sidebar.slider("Seuil de volume (%)", 50, 200, 100)

        # Paramètres de signal
        st.sidebar.write("**Paramètres de signal**")
        signal_threshold = st.sidebar.slider("Seuil de signal", 1, 4, 2)

        # Génération des signaux
        signals = pd.DataFrame(index=df.index)

        # Signal RSI
        signals['RSI_Signal'] = np.where(df['RSI'] < rsi_oversold, 1, np.where(df['RSI'] > rsi_overbought, -1, 0))

        # Signal MACD
        signals['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)

        # Signal Bollinger Bands
        signals['BB_Signal'] = np.where(df['close'] < df['BB_Lower'], 1, np.where(df['close'] > df['BB_Upper'], -1, 0))

        # Signal Volume
        volume_ma = df['volume'].rolling(window=volume_ma_period).mean()
        signals['Volume_Signal'] = np.where(df['volume'] > volume_ma * (volume_threshold/100), 1, 0)

        # Signal combiné
        signals['Combined_Signal'] = signals['RSI_Signal'] + signals['MACD_Signal'] + signals['BB_Signal'] + signals['Volume_Signal']

        # Simulation de trading
        initial_capital = 10000
        position = 0
        capital = initial_capital
        trades = []
        entry_price = 0

        for i in range(len(df)):
            current_price = df['close'].iloc[i]

            if position == 0:  # Pas de position ouverte
                if signals['Combined_Signal'].iloc[i] >= signal_threshold:  # Signal d'achat
                    position = capital / current_price
                    entry_price = current_price
                    trades.append({
                        'date': df.index[i],
                        'type': 'buy',
                        'price': current_price,
                        'position': position,
                        'capital': capital,
                        'signal_strength': signals['Combined_Signal'].iloc[i]
                    })
            else:  # Position ouverte
                # Vérification des conditions de sortie
                price_change = (current_price - entry_price) / entry_price * 100

                if (signals['Combined_Signal'].iloc[i] <= -signal_threshold or  # Signal de vente
                    price_change <= -stop_loss or  # Stop loss
                    price_change >= take_profit):  # Take profit

                    capital = position * current_price
                    trades.append({
                        'date': df.index[i],
                        'type': 'sell',
                        'price': current_price,
                        'position': position,
                        'capital': capital,
                        'profit_pct': price_change,
                        'exit_reason': 'Signal' if signals['Combined_Signal'].iloc[i] <= -signal_threshold else 
                                     'Stop Loss' if price_change <= -stop_loss else 'Take Profit'
                    })
                    position = 0

        # Calcul des métriques
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            # Calcul des métriques de performance
            total_trades = len(trades_df[trades_df['type'] == 'sell'])
            winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
            losing_trades = len(trades_df[trades_df['profit_pct'] <= 0])

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_profit = trades_df[trades_df['profit_pct'] > 0]['profit_pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['profit_pct'] <= 0]['profit_pct'].mean() if losing_trades > 0 else 0

            final_capital = trades_df['capital'].iloc[-1] if len(trades_df) > 0 else initial_capital
            total_return = (final_capital - initial_capital) / initial_capital * 100

            # Calcul des métriques supplémentaires
            trades_df['holding_period'] = trades_df.groupby((trades_df['type'] == 'buy').cumsum())['date'].transform(
                lambda x: (x.max() - x.min()).days
            )
            avg_holding_period = trades_df[trades_df['type'] == 'sell']['holding_period'].mean()

            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate * 100,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_return': total_return,
                'final_capital': final_capital,
                'avg_holding_period': avg_holding_period
            }

            # Affichage des résultats
            st.subheader("Résultats du Backtest")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Capital final", f"${metrics['final_capital']:,.2f}")
                st.metric("Rendement total", f"{metrics['total_return']:,.2f}%")
            with col2:
                st.metric("Trades gagnants", f"{metrics['winning_trades']}")
                st.metric("Win rate", f"{metrics['win_rate']:,.2f}%")
            with col3:
                st.metric("Profit moyen", f"{metrics['avg_profit']:,.2f}%")
                st.metric("Perte moyenne", f"{metrics['avg_loss']:,.2f}%")

            # Graphique des trades
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Prix', line=dict(color='blue')))

            for trade in trades_df.itertuples():
                if trade.type == 'buy':
                    fig.add_trace(go.Scatter(
                        x=[trade.date],
                        y=[trade.price],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name='Achat'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[trade.date],
                        y=[trade.price],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name='Vente'
                    ))

            fig.update_layout(
                title='Trades sur le prix du Bitcoin',
                yaxis_title='Prix (USD)',
                xaxis_title='Date',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tableau des derniers trades
            st.subheader("Derniers trades")
            trades_df['profit_pct'] = trades_df['profit_pct'].round(2)
            trades_df['holding_period'] = trades_df['holding_period'].round(1)
            st.dataframe(trades_df.tail(10))

            # Analyse des sorties de position
            st.subheader("Analyse des sorties de position")
            exit_reasons = trades_df[trades_df['type'] == 'sell']['exit_reason'].value_counts()
            fig_exit = go.Figure(data=[go.Pie(
                labels=exit_reasons.index,
                values=exit_reasons.values,
                hole=.3
            )])
            fig_exit.update_layout(title='Raisons de sortie des positions')
            st.plotly_chart(fig_exit, use_container_width=True)
        else:
            st.warning("Aucun trade n'a été effectué pendant cette période. Essayez d'ajuster les paramètres de la stratégie.")

    elif onglet == "Advanced Strategies":
        st.subheader("📊 Stratégies Avancées de Trading")

        # Sélection de la stratégie
        strategy = st.sidebar.selectbox(
            "Choisir une stratégie",
            ["Croisement des Moyennes Mobiles", "Stratégie de Volatilité", "Stratégie de Volume", "Fear & Greed"]
        )

        # Métriques communes pour toutes les stratégies
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            current_price = df['close'].iloc[-1]
            st.metric("Prix actuel", f"${current_price:,.2f}")
        with col2:
            daily_change = df['close'].pct_change().iloc[-1] * 100
            st.metric("Variation journalière", f"{daily_change:+.2f}%", 
                     delta_color="normal" if daily_change >= 0 else "inverse")
        with col3:
            volatility = df['returns'].rolling(window=30).std().iloc[-1] * 100
            st.metric("Volatilité (30j)", f"{volatility:.2f}%")
        with col4:
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            st.metric("Ratio de volume", f"{volume_ratio:.2f}x")

        if strategy == "Croisement des Moyennes Mobiles":
            st.write("### 📈 Stratégie de Croisement des Moyennes Mobiles")

            # Description de la stratégie
            st.markdown("""
            **Description de la stratégie :**
            - Utilise les croisements entre les moyennes mobiles de 20, 50 et 200 jours
            - Signal d'achat : croisement à la hausse
            - Signal de vente : croisement à la baisse
            - Plus fiable en tendance forte
            """)

            # Paramètres ajustables
            st.sidebar.subheader("Paramètres")
            sma_short = st.sidebar.slider("Période courte", 5, 50, 20)
            sma_medium = st.sidebar.slider("Période moyenne", 30, 100, 50)
            sma_long = st.sidebar.slider("Période longue", 100, 300, 200)

            # Calcul des signaux
            df = compute_moving_average_crossover(df)

            # Affichage des signaux
            st.write("#### 📊 Signaux de Trading")
            signals = pd.DataFrame(index=df.index)
            signals['Signal 20-50'] = np.where(df['Cross_20_50'] == 2, '🟢 Achat', 
                                             np.where(df['Cross_20_50'] == -2, '🔴 Vente', '⚪ Neutre'))
            signals['Signal 50-200'] = np.where(df['Cross_50_200'] == 2, '🟢 Achat', 
                                              np.where(df['Cross_50_200'] == -2, '🔴 Vente', '⚪ Neutre'))

            # Statistiques des signaux
            st.write("#### 📈 Statistiques des Signaux")
            col1, col2 = st.columns(2)
            with col1:
                signal_stats = pd.DataFrame({
                    'Signal': ['Achat', 'Vente', 'Neutre'],
                    '20-50': signals['Signal 20-50'].value_counts(),
                    '50-200': signals['Signal 50-200'].value_counts()
                })
                st.dataframe(signal_stats)

            with col2:
                # Performance des signaux
                signal_returns = pd.DataFrame({
                    'Signal': signals['Signal 20-50'],
                    'Return': df['returns'] * 100
                })
                performance = signal_returns.groupby('Signal')['Return'].agg(['mean', 'std', 'count'])
                st.dataframe(performance)

            # Graphique principal
            fig = make_subplots(rows=2, cols=1, 
                               shared_xaxes=True,
                               vertical_spacing=0.03,
                               row_heights=[0.7, 0.3])

            # Prix et moyennes mobiles
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Prix', line=dict(color='white')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='red')), row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', 
                                marker_color='rgba(0, 255, 0, 0.3)'), row=2, col=1)

            fig.update_layout(
                title='Prix et Moyennes Mobiles',
                height=800,
                template='plotly_dark',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Derniers signaux
            st.write("#### 📅 Derniers Signaux")
            st.dataframe(signals.tail(10))

        elif strategy == "Stratégie de Volatilité":
            st.write("### 📊 Stratégie de Volatilité")

            # Description de la stratégie
            st.markdown("""
            **Description de la stratégie :**
            - Utilise les bandes de Bollinger et le RSI
            - Signal d'achat : prix sous la bande inférieure + RSI < 30
            - Signal de vente : prix au-dessus de la bande supérieure + RSI > 70
            - Idéal pour les marchés en range
            """)

            # Paramètres ajustables
            st.sidebar.subheader("Paramètres")
            bb_period = st.sidebar.slider("Période BB", 10, 50, 20)
            bb_std = st.sidebar.slider("Écart-type BB", 1.0, 3.0, 2.0)
            rsi_period = st.sidebar.slider("Période RSI", 5, 30, 14)

            # Calcul des signaux
            df = compute_volatility_strategy(df)

            # Affichage des signaux
            st.write("#### 📊 Signaux de Trading")
            signals = pd.DataFrame(index=df.index)
            signals['Signal'] = np.where(df['Volatility_Signal'] == 1, '🟢 Achat', 
                                       np.where(df['Volatility_Signal'] == -1, '🔴 Vente', '⚪ Neutre'))

            # Statistiques des signaux
            st.write("#### 📈 Statistiques des Signaux")
            col1, col2 = st.columns(2)
            with col1:
                signal_stats = signals['Signal'].value_counts()
                st.dataframe(signal_stats)

            with col2:
                # Performance des signaux
                signal_returns = pd.DataFrame({
                    'Signal': signals['Signal'],
                    'Return': df['returns'] * 100
                })
                performance = signal_returns.groupby('Signal')['Return'].agg(['mean', 'std', 'count'])
                st.dataframe(performance)

            # Graphique principal
            fig = make_subplots(rows=3, cols=1, 
                               shared_xaxes=True,
                               vertical_spacing=0.03,
                               row_heights=[0.5, 0.25, 0.25])

            # Prix et bandes de Bollinger
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Prix', line=dict(color='white')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Sup', 
                                    line=dict(color='red', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Inf', 
                                    line=dict(color='green', dash='dash')), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', 
                                marker_color='rgba(0, 255, 0, 0.3)'), row=3, col=1)

            fig.update_layout(
                title='Analyse de Volatilité',
                height=1000,
                template='plotly_dark',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Derniers signaux
            st.write("#### 📅 Derniers Signaux")
            st.dataframe(signals.tail(10))

        elif strategy == "Stratégie de Volume":
            st.write("### 📊 Stratégie de Volume")

            # Description de la stratégie
            st.markdown("""
            **Description de la stratégie :**
            - Utilise l'OBV (On-Balance Volume) et le MFI (Money Flow Index)
            - Signal d'achat : OBV en hausse + MFI < 20
            - Signal de vente : OBV en baisse + MFI > 80
            - Confirme les mouvements de prix
            """)

            # Paramètres ajustables
            st.sidebar.subheader("Paramètres")
            mfi_period = st.sidebar.slider("Période MFI", 5, 30, 14)
            obv_threshold = st.sidebar.slider("Seuil OBV", 0.5, 2.0, 1.0)

            # Calcul des signaux
            df = compute_volume_strategy(df)

            # Affichage des signaux
            st.write("#### 📊 Signaux de Trading")
            signals = pd.DataFrame(index=df.index)
            signals['Signal'] = np.where(df['Volume_Signal'] == 1, '🟢 Achat', 
                                       np.where(df['Volume_Signal'] == -1, '🔴 Vente', '⚪ Neutre'))

            # Statistiques des signaux
            st.write("#### 📈 Statistiques des Signaux")
            col1, col2 = st.columns(2)
            with col1:
                signal_stats = signals['Signal'].value_counts()
                st.dataframe(signal_stats)

            with col2:
                # Performance des signaux
                signal_returns = pd.DataFrame({
                    'Signal': signals['Signal'],
                    'Return': df['returns'] * 100
                })
                performance = signal_returns.groupby('Signal')['Return'].agg(['mean', 'std', 'count'])
                st.dataframe(performance)

            # Graphique principal
            fig = make_subplots(rows=3, cols=1, 
                               shared_xaxes=True,
                               vertical_spacing=0.03,
                               row_heights=[0.4, 0.3, 0.3])

            # Prix
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Prix', line=dict(color='white')), row=1, col=1)

            # OBV
            fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', line=dict(color='blue')), row=2, col=1)

            # MFI
            fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], name='MFI'), row=3, col=1)
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)

            fig.update_layout(
                title='Analyse de Volume',
                height=1000,
                template='plotly_dark',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Derniers signaux
            st.write("#### 📅 Derniers Signaux")
            st.dataframe(signals.tail(10))

        elif strategy == "Fear & Greed":
            st.write("### 😱 Fear & Greed Index")

            # Description de la stratégie
            st.markdown("""
            **Description de l'indice :**
            - Mesure la psychologie du marché
            - 0-20 : Extrême Peur (potentiel d'achat)
            - 80-100 : Extrême Greed (potentiel de vente)
            - 40-60 : Marché neutre
            """)

            # Récupération de l'indice
            fear_greed = get_fear_greed_index()

            # Affichage de l'indice
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fear & Greed Index", f"{fear_greed}")
            with col2:
                if fear_greed <= 20:
                    st.success("Extrême Peur - Potentiel d'achat")
                elif fear_greed >= 80:
                    st.warning("Extrême Greed - Potentiel de vente")
                else:
                    st.info("Marché neutre")
            with col3:
                st.metric("Tendance", "Haussière" if fear_greed < 50 else "Baissière")

            # Graphique de l'indice
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=fear_greed,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "green"},
                        {'range': [20, 40], 'color': "lightgreen"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': fear_greed
                    }
                }
            ))
            fig.update_layout(
                title='Fear & Greed Index',
                height=400,
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Historique des signaux
            st.write("#### 📅 Historique des Signaux")
            signals = pd.DataFrame({
                'Date': df.index[-10:],
                'Prix': df['close'].iloc[-10:],
                'Variation': df['returns'].iloc[-10:] * 100,
                'Volume': df['volume'].iloc[-10:],
                'RSI': df['RSI'].iloc[-10:],
                'MACD': df['MACD'].iloc[-10:]
            })
            st.dataframe(signals)

if __name__ == "__main__":
    main()