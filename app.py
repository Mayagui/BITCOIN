import streamlit as st
# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="Bitcoin Analytics",
    page_icon="₿",
    layout="wide"
)

import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from datetime import datetime, timedelta
import pytz
import sqlite3
from pytrends.request import TrendReq
from streamlit_autorefresh import st_autorefresh
import hashlib
from textblob import TextBlob
import newspaper
import os
import snscrape.modules.twitter as sntwitter

# Rafraîchissement automatique toutes les 60 secondes (au lieu de 10)
st_autorefresh(interval=60 * 1000, key="refresh")

st.write("Début du script")

# Style personnalisé
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

class BitcoinAnalyzer:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000
        })
        self.last_data = None
        self.last_update = None
    
    def get_current_price(self):
        try:
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            return {
                'price': ticker['last'],
                'volume': ticker['quoteVolume'],
                'change': ticker['percentage']
            }
        except Exception as e:
            st.error(f"Erreur lors de la récupération du prix actuel : {str(e)}")
            return None
    
    def get_historical_data(self, timeframe='1d', start_date=None, end_date=None):
        try:
            if start_date is None or end_date is None:
                end_date = datetime.now(pytz.UTC)
                start_date = end_date - timedelta(days=365)
            since = int(start_date.timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(
                symbol='BTC/USDT',
                timeframe=timeframe,
                since=since,
                limit=1500
            )
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            # Conversion explicite
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
            end_date_pd = pd.Timestamp(end_date).tz_localize(None)
            df = df[df['timestamp'] <= end_date_pd]
            return df
        except Exception as e:
            st.error(f"Erreur lors de la récupération des données : {str(e)}")
            return None

class DatabaseManager:
    def __init__(self):
        # Stockage dans le dossier de l'application
        self.db_path = 'bitcoin_data.db'
        self.init_database()
    
    def init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Création de la table si elle n'existe pas
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bitcoin_prices (
                timestamp DATETIME PRIMARY KEY,
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
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql('bitcoin_prices', conn, if_exists='append', index=False)
            conn.close()
        except Exception as e:
            st.error(f"Erreur de sauvegarde en masse: {str(e)}")
    
    def get_data(self, days=30):
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

def fetch_historical_data(symbol="BTC/USDT", timeframe="1d", since_days=365*5):
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=since_days)).strftime('%Y-%m-%dT%H:%M:%S'))
    all_ohlcv = []
    limit = 1000
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def display_single_metric(container, label, value, delta=None, delta_color=None):
    with container:
        if delta_color == "positive":
            delta_prefix = "↑ "
        elif delta_color == "negative":
            delta_prefix = "↓ "
        else:
            delta_prefix = ""
            
        st.metric(
            label=label,
            value=value,
            delta=f"{delta_prefix}{delta}" if delta else None
        )

def add_export_button(df):
    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Exporter les données en CSV",
            data=csv,
            file_name=f'bitcoin_data_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv'
        )

def create_price_chart(df):
    fig = make_subplots(
        rows=4,  # Passe à 4 lignes
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('BTC/USDT', 'RSI', 'MACD', 'Volume')
    )

    # Chandeliers
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

    # Moyennes mobiles
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

    # RSI
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

    # MACD
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

    # Volume
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
    """
    Récupère l'intérêt Google Trends pour un mot-clé.
    Retourne le score moyen sur la période.
    """
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

@st.cache_data(ttl=3600)  # Cache la donnée pendant 1h
def get_google_trends_score_cached(keyword="Bitcoin", timeframe="now 7-d", geo=""):
    return get_google_trends_score(keyword, timeframe, geo)

def get_google_news_sentiment(query="Bitcoin", lang="fr"):
    """
    Analyse de sentiment sur les titres Google News pour un mot-clé.
    Retourne le score moyen de sentiment.
    """
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

def get_twitter_sentiment(query="Bitcoin", limit=20):
    sentiments = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{query} lang:fr').get_items()):
        if i >= limit:
            break
        blob = TextBlob(tweet.content)
        sentiments.append(blob.sentiment.polarity)
    if sentiments:
        return sum(sentiments) / len(sentiments)
    else:
        return None

def plot_trends(keyword="Bitcoin", timeframe="now 7-d"):
    pytrends = TrendReq(hl='fr-FR', tz=360)
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe)
    data = pytrends.interest_over_time()
    if not data.empty:
        st.line_chart(data[keyword])
    else:
        st.info("Pas de données Google Trends pour cette période.")

def period_to_days(period_code):
    if period_code == "1m":
        return 30
    elif period_code == "3m":
        return 90
    elif period_code == "6m":
        return 180
    elif period_code == "1y":
        return 365
    elif period_code == "2y":
        return 365*2
    elif period_code == "3y":
        return 365*3
    elif period_code == "4y":
        return 365*4
    elif period_code == "5y":
        return 365*5
    else:
        return 365  # Par défaut 1 an

def compute_global_score(rsi, macd, macd_signal, price_change, sentiment, trends_score):
    score = 0
    # RSI : suracheté (<30) ou survendu (>70)
    if rsi < 30:
        score += 1
    elif rsi > 70:
        score -= 1

    # MACD croise au-dessus du signal = haussier
    if macd > macd_signal:
        score += 1
    else:
        score -= 1

    # Variation du prix sur 24h
    if price_change > 0:
        score += 1
    else:
        score -= 1

    # Sentiment Google News
    if sentiment is not None:
        if sentiment > 0.1:
            score += 1
        elif sentiment < -0.1:
            score -= 1

    # Google Trends
    if trends_score is not None:
        if trends_score > 50:
            score += 1
        else:
            score -= 1

    # Score global sur 5
    return score

def detect_rsi_divergence(df):
    # Divergence haussière : prix fait un plus bas, RSI fait un plus haut
    if len(df) < 20:
        return None
    price = df['close']
    rsi = ta.momentum.rsi(df['close'])
    # Compare les 10 derniers points
    if price.iloc[-1] < price.iloc[-10] and rsi.iloc[-1] > rsi.iloc[-10]:
        return "Divergence haussière"
    elif price.iloc[-1] > price.iloc[-10] and rsi.iloc[-1] < rsi.iloc[-10]:
        return "Divergence baissière"
    else:
        return None

def backtest_strategy(df, score_func, threshold=1):
    """
    Backtest simple : achat si score > threshold, vente sinon.
    On suppose qu'on est toujours investi ou non (pas de levier, pas de frais).
    """
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['trends_score'], _ = get_google_trends_score_cached("Bitcoin", "now 7-d")
    df['news_sentiment'] = get_google_news_sentiment("Bitcoin")
    df['score'] = df.apply(
        lambda row: score_func(
            row['rsi'],
            row['macd'],
            row['macd_signal'],
            row['close'].pct_change() * 100 if not pd.isna(row['close']) else 0,
            row['news_sentiment'],
            row['trends_score']
        ), axis=1
    )
    df['signal'] = df['score'] > threshold
    df['btc_ret'] = df['close'].pct_change()
    df['strategy_ret'] = df['btc_ret'] * df['signal'].shift(1).fillna(0)
    df['equity'] = (1 + df['strategy_ret']).cumprod()
    return df

def main():
    analyzer = BitcoinAnalyzer()
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

    if st.button("Importer l'historique BTC/USDT (1 an)"):
        df_hist = fetch_historical_data("BTC/USDT", "1d", since_days=365)
        db.save_bulk_data(df_hist)
        st.success(f"{len(df_hist)} lignes importées dans la base.")

    if st.button("Importer l'historique BTC/USDT (5 ans)"):
        df_hist_5y = fetch_historical_data("BTC/USDT", "1d", since_days=365*5)
        db.save_bulk_data(df_hist_5y)
        st.success(f"{len(df_hist_5y)} lignes importées dans la base (5 ans).")

    if st.button("Mettre à jour le CSV (5 ans)"):
        with st.spinner("Téléchargement en cours..."):
            df_hist_5y = fetch_historical_data("BTC/USDT", "1d", since_days=365*5)
            df_hist_5y.to_csv("bitcoin_5y.csv", index=False)
            st.success(f"{len(df_hist_5y)} lignes sauvegardées dans bitcoin_5y.csv")

    # --- Chargement rapide de l'historique si période longue ---
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
        df = analyzer.get_historical_data(timeframe[1], start_date, end_date)

    # --- Affichage des métriques et du graphique (toujours affichés) ---
    if df is not None and not df.empty:
        current_data = analyzer.get_current_price()
        if current_data:
            rsi = ta.momentum.rsi(df['close']).iloc[-1]
            macd_val = ta.trend.macd(df['close']).iloc[-1]
            data_to_save = {
                'price': current_data['price'],
                'volume': current_data['volume'],
                'rsi': rsi,
                'change': current_data['change']
            }
            db.save_data(data_to_save)

            cols = st.columns(4)
            with cols[0]:
                st.metric("PRIX BTC", f"${current_data['price']:,.2f}", f"{current_data['change']:.2f}%")
            with cols[1]:
                st.metric("RSI", f"{rsi:.1f}", "Suracheté" if rsi > 70 else "Survendu" if rsi < 30 else "Neutre")
            with cols[2]:
                st.metric("Volume 24h", f"${current_data['volume']/1e9:.2f}B")
            with cols[3]:
                st.metric("Tendance", "🟢 Haussière" if current_data['change'] > 0 else "🔴 Baissière")
            st.metric("MACD", f"{macd_val:.2f}")

            # Détection de divergence RSI
            divergence = detect_rsi_divergence(df)
            if divergence:
                st.warning(f"⚠️ {divergence} détectée (RSI)")

            # Google Trends
            trends_score, trends_series = get_google_trends_score_cached("Bitcoin", "now 7-d")
            if trends_score is not None:
                st.metric("Tendance Google Trends", f"{trends_score}/100")

            # Google News Sentiment
            news_sentiment = get_google_news_sentiment("Bitcoin")
            if news_sentiment is not None:
                sentiment_label = "Positif" if news_sentiment > 0.1 else "Négatif" if news_sentiment < -0.1 else "Neutre"
                st.metric("Sentiment Google News", f"{news_sentiment:.2f}", sentiment_label)
            else:
                st.info("Pas de données Google News pour le sentiment.")

            # Twitter Sentiment
            try:
                twitter_sentiment = get_twitter_sentiment("Bitcoin")
                if twitter_sentiment is not None:
                    sentiment_label = "Positif" if twitter_sentiment > 0.1 else "Négatif" if twitter_sentiment < -0.1 else "Neutre"
                    st.metric("Sentiment Twitter", f"{twitter_sentiment:.2f}", sentiment_label)
                else:
                    st.info("Pas de données Twitter pour le sentiment.")
            except Exception as e:
                st.warning(f"Erreur Twitter : {e}")

            # Score Global
            macd_signal = ta.trend.macd_signal(df['close']).iloc[-1]
            global_score = compute_global_score(
                rsi,
                macd_val,
                macd_signal,
                current_data['change'],
                news_sentiment,
                trends_score
            )
            score_label = "Haussier" if global_score > 1 else "Baissier" if global_score < -1 else "Neutre"

            # Volume Median Check
            volume_median = df['volume'].tail(30).median()
            if current_data['volume'] > volume_median:
                st.metric("Score Global", f"{global_score}/5", score_label)
            else:
                st.info("Signal ignoré (volume trop faible)")

        fig = create_price_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune donnée à afficher.")

    # --- Backtesting ---
    if st.button("Lancer le backtesting sur la période affichée"):
        with st.spinner("Backtesting en cours..."):
            df_bt = backtest_strategy(df, compute_global_score)
            st.line_chart(df_bt.set_index('timestamp')['equity'])
            perf = (df_bt['equity'].iloc[-1] - 1) * 100
            st.success(f"Performance de la stratégie : {perf:.2f}%")
            st.info(f"Nombre de signaux d'achat : {df_bt['signal'].sum()}")

    # --- Affichage de l'historique stocké (optionnel, toujours en dehors du bloc principal) ---
    if show_stored_data:
        stored_data = db.get_data(days_to_show)
        if not stored_data.empty:
            import hashlib
            df_hash = hashlib.md5(pd.util.hash_pandas_object(stored_data, index=True).values).hexdigest()
            with st.expander("📊 Historique des prix stockés", expanded=True):
                st.dataframe(stored_data)
                csv = stored_data.to_csv(index=False)
                st.download_button(
                    label="📥 Exporter en CSV",
                    data=csv,
                    file_name=f'bitcoin_data_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                    mime='text/csv',
                    key="download-csv-stock"
                )

if __name__ == "__main__":
    main()