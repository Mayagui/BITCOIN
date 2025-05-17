import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from datetime import datetime, timedelta
import pytz
import sqlite3
import os
import yfinance as yf

# Gestion des imports optionnels
missing_modules = []
try:
    from pytrends.request import TrendReq
except ImportError:
    missing_modules.append("pytrends")
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    missing_modules.append("streamlit_autorefresh")
try:
    from textblob import TextBlob
except ImportError:
    missing_modules.append("textblob")
try:
    import newspaper
except ImportError:
    missing_modules.append("newspaper")
try:
    from ml_utils import load_ml_model, predict_next_day_price
except ImportError:
    missing_modules.append("ml_utils")

if missing_modules:
    st.error(f"Modules manquants : {', '.join(missing_modules)}. Veuillez les installer pour profiter de toutes les fonctionnalités.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# Rafraîchissement automatique toutes les 60 secondes
if "streamlit_autorefresh" not in missing_modules:
    st_autorefresh(interval=60 * 1000, key="refresh")

st.set_page_config(
    page_title="Bitcoin Analytics",
    page_icon="₿",
    layout="wide"
)

# --- CSS custom ---
def custom_css():
    """Ajoute du CSS personnalisé à l'application Streamlit."""
    st.markdown("""
    <style>
        .stMetric {
            background-color: #1E2126;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #2D3035;
        }
        .stMetric:hover {
            border-color: #4A4E54;
        }
        .metric-label {
            font-size: 0.8rem !important;
            color: #7C7C7C !important;
        }
        .metric-value {
            font-size: 1.8rem !important;
            font-weight: bold !important;
        }
        .metric-delta {
            font-size: 1rem !important;
        }
        .positive {
            color: #00FF9F !important;
        }
        .negative {
            color: #FF4B4B !important;
        }
        .neutral {
            color: #7C7C7C !important;
        }
    </style>
    """, unsafe_allow_html=True)

custom_css()

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
                price FLOAT,
                volume FLOAT,
                rsi FLOAT,
                change_24h FLOAT
            )
            ''')
            conn.commit()
            conn.close()
            st.sidebar.success("Base de données initialisée avec succès")
        except Exception as e:
            st.sidebar.error(f"Erreur d'initialisation de la base de données: {str(e)}")
    
    def save_data(self, data):
        """Sauvegarde une ligne de données dans la base."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO bitcoin_prices 
            (timestamp, price, volume, rsi, change_24h)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                data['price'],
                data['volume'],
                data['rsi'],
                data['change']
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            st.error(f"Erreur de sauvegarde: {str(e)}")
    
    def save_bulk_data(self, df):
        """Sauvegarde un DataFrame entier dans la base."""
        try:
            conn = sqlite3.connect(self.db_path)
            expected_cols = [
                'timestamp', 'open', 'high', 'low', 'close', 'price', 'volume', 'rsi', 'change_24h'
            ]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
            df = df[expected_cols]
            df.to_sql('bitcoin_prices', conn, if_exists='append', index=False)
            conn.close()
        except Exception as e:
            st.error(f"Erreur de sauvegarde en masse: {str(e)}")
    
    def get_data(self, days=30):
        """Récupère les données stockées sur X jours."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
            SELECT * FROM bitcoin_prices 
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(days,))
            conn.close()
            return df
        except Exception as e:
            st.error(f"Erreur de lecture: {str(e)}")
            return pd.DataFrame()

def get_btc_usdt_history(period="1y", interval="1d"):
    """Télécharge l'historique BTC-USD depuis Yahoo Finance."""
    try:
        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(period=period, interval=interval)
        df = df.reset_index()
        df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données Yahoo Finance : {e}")
        return pd.DataFrame()

def add_export_button(df):
    """Ajoute un bouton d'export CSV."""
    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Exporter les données en CSV",
            data=csv,
            file_name=f'bitcoin_data_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv'
        )

def create_price_chart(df):
    """Crée le graphique principal avec Plotly."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('BTC/USDT', 'RSI', 'MACD', 'Volume')
    )
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTC/USDT'
        ),
        row=1, col=1
    )
    for window, color in zip([50, 100, 200], ['red', 'green', 'purple']):
        ma = ta.trend.sma_indicator(df['close'], window=window)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=ma,
                name=f'SMA {window}',
                line=dict(color=color, width=1)
            ),
            row=1, col=1
        )
    rsi = ta.momentum.rsi(df['close'])
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=rsi,
            name='RSI',
            line=dict(color='blue', width=1)
        ),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2)
    macd = ta.trend.macd(df['close'])
    macd_signal = ta.trend.macd_signal(df['close'])
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=macd,
            name='MACD',
            line=dict(color='orange', width=1)
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=macd_signal,
            name='Signal MACD',
            line=dict(color='green', width=1, dash='dot')
        ),
        row=3, col=1
    )
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=4, col=1
    )
    fig.update_layout(
        height=1000,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def get_google_trends_score(keyword="Bitcoin", timeframe="now 7-d", geo=""):
    """Récupère l'intérêt Google Trends pour un mot-clé."""
    if "pytrends" in missing_modules:
        return None, None
    try:
        pytrends = TrendReq(hl='fr-FR', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
        data = pytrends.interest_over_time()
        if not data.empty:
            score = int(data[keyword].mean())
            return score, data[keyword]
        else:
            return None, None
    except Exception as e:
        st.warning(f"Erreur Google Trends : {e}")
        return None, None

@st.cache_data(ttl=3600)
def get_google_trends_score_cached(keyword="Bitcoin", timeframe="now 7-d", geo=""):
    return get_google_trends_score(keyword, timeframe, geo)

def get_google_news_sentiment(query="Bitcoin", lang="fr"):
    """Analyse de sentiment sur les titres Google News pour un mot-clé."""
    if "newspaper" in missing_modules or "textblob" in missing_modules:
        return None
    try:
        url = f"https://news.google.com/rss/search?q={query}&hl={lang}"
        paper = newspaper.build(url, memoize_articles=False)
        sentiments = []
        for article in paper.articles[:10]:
            try:
                article.download()
                article.parse()
                blob = TextBlob(article.title)
                sentiments.append(blob.sentiment.polarity)
            except Exception:
                continue
        if sentiments:
            return sum(sentiments) / len(sentiments)
        else:
            return None
    except Exception as e:
        st.warning(f"Erreur Google News : {e}")
        return None

def compute_global_score(rsi, macd, macd_signal, price_change, sentiment, trends_score):
    """Calcule un score global synthétique."""
    score = 0
    if rsi < 30:
        score += 1
    elif rsi > 70:
        score -= 1
    if macd > macd_signal:
        score += 1
    else:
        score -= 1
    if price_change > 0:
        score += 1
    else:
        score -= 1
    if sentiment is not None:
        if sentiment > 0.1:
            score += 1
        elif sentiment < -0.1:
            score -= 1
    if trends_score is not None:
        if trends_score > 50:
            score += 1
        else:
            score -= 1
    return score

def detect_rsi_divergence(df):
    """Détecte une divergence RSI sur les 10 derniers points."""
    if len(df) < 20:
        return None
    price = df['close']
    rsi = ta.momentum.rsi(df['close'])
    if price.iloc[-1] < price.iloc[-10] and rsi.iloc[-1] > rsi.iloc[-10]:
        return "Divergence haussière"
    elif price.iloc[-1] > price.iloc[-10] and rsi.iloc[-1] < rsi.iloc[-10]:
        return "Divergence baissière"
    else:
        return None

def period_to_days(period_code):
    """Convertit un code période en nombre de jours."""
    mapping = {
        "1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730,
        "3y": 1095, "4y": 1460, "5y": 1825
    }
    return mapping.get(period_code, 365)

def main():
    """Point d'entrée principal de l'application Streamlit."""
    db = DatabaseManager()
    with st.sidebar:
        st.title("Configuration")
        timeframe = st.selectbox(
            "Intervalle",
            options=[
                ("Journalier", "1d"),
                ("4 Heures", "4h"),
                ("1 Heure", "1h")
            ],
            format_func=lambda x: x[0]
        )
        period_label = st.selectbox(
            "Période d'analyse",
            options=[
                ("1 mois", "1m"),
                ("3 mois", "3m"),
                ("6 mois", "6m"),
                ("1 an", "1y"),
                ("2 ans", "2y"),
                ("3 ans", "3y"),
                ("4 ans", "4y"),
                ("5 ans", "5y"),
            ],
            format_func=lambda x: x[0]
        )
        show_stored_data = st.checkbox("Afficher données stockées", value=True)
        days_to_show = st.slider("Historique à afficher (jours)", 1, 30, 7)

    # Vérification des fichiers nécessaires
    required_files = ["bitcoin_5y.csv", "ml_model.pkl"]
    for file in required_files:
        if os.path.exists(file):
            st.success(f"✅ Fichier présent : {file}")
        else:
            st.error(f"❌ Fichier manquant : {file}")

    # Importation de l'historique
    if st.button("Importer l'historique BTC/USDT (1 an)"):
        df_hist = get_btc_usdt_history(period="1y", interval="1d")
        db.save_bulk_data(df_hist)
        st.success(f"{len(df_hist)} lignes importées dans la base.")

    if st.button("Importer l'historique BTC/USDT (5 ans)"):
        df_hist_5y = get_btc_usdt_history(period="5y", interval="1d")
        db.save_bulk_data(df_hist_5y)
        st.success(f"{len(df_hist_5y)} lignes importées dans la base (5 ans).")

    if st.button("Mettre à jour le CSV (5 ans)"):
        with st.spinner("Téléchargement en cours..."):
            df_hist_5y = get_btc_usdt_history(period="5y", interval="1d")
            df_hist_5y.to_csv("bitcoin_5y.csv", index=False)
            st.success(f"{len(df_hist_5y)} lignes sauvegardées dans bitcoin_5y.csv")

    # Chargement rapide de l'historique si période longue
    csv_path = "bitcoin_5y.csv"
    use_csv = os.path.exists(csv_path) and period_label[1] in ["3y", "4y", "5y"]

    if use_csv:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        period_days = period_to_days(period_label[1])
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=period_days)
        df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
    else:
        period_days = period_to_days(period_label[1])
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=period_days)
        df = get_btc_usdt_history(period="1y", interval="1d")

    # Affichage des métriques et du graphique
    if df is not None and not df.empty:
        current_data = {
            'price': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1],
            'change': df['close'].pct_change().iloc[-1] * 100
        }
        if current_data:
            rsi = ta.momentum.rsi(df['close']).iloc[-1]
            macd_val = ta.trend.macd(df['close']).iloc[-1]
            macd_signal = ta.trend.macd_signal(df['close']).iloc[-1]
            price_change = current_data['change']
            sentiment = get_google_news_sentiment("Bitcoin")
            trends_score, _ = get_google_trends_score_cached("Bitcoin", "now 7-d")
            global_score = compute_global_score(rsi, macd_val, macd_signal, price_change, sentiment, trends_score)

            cols = st.columns(5)
            with cols[0]:
                st.metric("PRIX BTC", f"${current_data['price']:,.2f}", f"{current_data['change']:.2f}%")
            with cols[1]:
                st.metric("RSI", f"{rsi:.1f}", "Suracheté" if rsi > 70 else "Survendu" if rsi < 30 else "Neutre")
            with cols[2]:
                st.metric("Volume 24h", f"${current_data['volume']/1e9:.2f}B")
            with cols[3]:
                st.metric("MACD", f"{macd_val:.2f}")
            with cols[4]:
                label = "🟢 Haussier" if global_score > 1 else "🔴 Baissier" if global_score < -1 else "⚪️ Neutre"
                st.metric("Score Global", f"{global_score}/5", label)

            with st.container():
                cols_sent = st.columns(2)
                with cols_sent[0]:
                    if sentiment is not None:
                        sentiment_label = "🟢 Positif" if sentiment > 0.1 else "🔴 Négatif" if sentiment < -0.1 else "⚪️ Neutre"
                        st.metric("Sentiment Google News", f"{sentiment:.2f}", sentiment_label)
                    else:
                        st.caption("Aucune donnée Google News disponible pour le sentiment.")
                with cols_sent[1]:
                    if trends_score is not None:
                        trends_label = "🟢 Fort intérêt" if trends_score > 50 else "🔴 Faible intérêt"
                        st.metric("Google Trends (7j)", f"{trends_score}", trends_label)
                    else:
                        st.caption("Aucune donnée Google Trends disponible.")

            data_to_save = {
                'price': current_data['price'],
                'volume': current_data['volume'],
                'rsi': rsi,
                'change': current_data['change']
            }
            db.save_data(data_to_save)

            if rsi > 70:
                st.warning("⚠️ RSI suracheté : risque de correction baissière.")
            elif rsi < 30:
                st.info("🔵 RSI survendu : possible rebond haussier.")
            else:
                st.info("RSI neutre.")

            if macd_val > macd_signal:
                st.success("MACD haussier : momentum positif.")
            else:
                st.error("MACD baissier : momentum négatif.")

            divergence = detect_rsi_divergence(df)
            if divergence:
                st.warning(f"⚠️ {divergence} détectée (RSI)")

        st.plotly_chart(create_price_chart(df), use_container_width=True)
        if show_stored_data:
            st.subheader("📊 Historique des prix stockés")
            df_display = df.copy()
            df_display['rsi'] = ta.momentum.rsi(df_display['close'])
            df_display['change_24h'] = df_display['close'].pct_change(periods=1) * 100
            st.dataframe(df_display.head(days_to_show))
            add_export_button(df_display.head(days_to_show))

        if st.button("Prédire la tendance ML (prochain jour)"):
            if "ml_utils" not in missing_modules:
                last_row = df.tail(1).copy()
                last_row['rsi'] = ta.momentum.rsi(df['close']).iloc[-1]
                last_row['macd'] = ta.trend.macd(df['close']).iloc[-1]
                last_row['macd_signal'] = ta.trend.macd_signal(df['close']).iloc[-1]
                last_row['volume'] = df['volume'].iloc[-1]
                model = load_ml_model("ml_model.pkl")
                pred, proba = predict_next_day_price(model, last_row)
                label = "🟢 Hausse probable" if pred == 1 else "🔴 Baisse probable"
                st.metric("Prédiction ML", label, f"Confiance : {proba:.2%}")
            else:
                st.error("Module ml_utils manquant pour la prédiction ML.")

if __name__ == "__main__":
    main()
