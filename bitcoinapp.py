import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data_collector import BitcoinDataCollector
from sentiment_analyzer import EnhancedSentimentAnalyzer
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import time
import ta.momentum
import os
import pickle

class BitcoinDashboard:
    def __init__(self):
        self.data_collector = BitcoinDataCollector()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        
    def run(self):
        st.title("Bitcoin Analysis Dashboard")
        
        # Sidebar pour les contrôles
        st.sidebar.header("Paramètres")
        timeframe = st.sidebar.selectbox(
            "Période d'analyse",
            ["1j", "1s", "1m", "3m", "6m", "1a", "5a"]
        )
        
        # Onglets principaux
        tab1, tab2, tab3 = st.tabs(["Prix & Indicateurs", "Analyse des Sentiments", "Données Historiques"])
        
        with tab1:
            self.display_price_indicators()
            
        with tab2:
            self.display_sentiment_analysis()
            
        with tab3:
            self.display_historical_data()
    
    def display_price_indicators(self):
        # Récupération des données en temps réel
        current_data = self.data_collector.realtime_collection.find().sort([("timestamp", -1)]).limit(1000)
        df = pd.DataFrame(list(current_data))
        if df is None or df.empty or not all(col in df.columns for col in ['close', 'open', 'high', 'low', 'volume', 'timestamp']):
            st.warning("Aucune donnée ou colonnes manquantes pour l'affichage.")
            return
        
        # Calcul du RSI
        rsi = ta.momentum.rsi(df['close']).iloc[-1]
    
        # Création du graphique
        fig = make_subplots(rows=2, cols=1, shared_xaxis=True)
        
        # Graphique des prix
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close']
            ),
            row=1, col=1
        )
        
        # Graphique du volume
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume']),
            row=2, col=1
        )
        
        st.plotly_chart(fig)
        
        # Google Trends
        trends_score, trends_series = get_google_trends_score_cached("Bitcoin", "now 7-d")

        cols = st.columns(5)
        with cols[0]:
            st.metric("PRIX BTC", f"${df['close'].iloc[-1]:,.2f}")
        with cols[1]:
            st.metric("RSI", f"{rsi:.1f}", "Suracheté" if rsi > 70 else "Survendu" if rsi < 30 else "Neutre")
        with cols[2]:
            st.metric("Volume 24h", f"${df['volume'].iloc[-1]/1e9:.2f}B")
        with cols[3]:
            change = df['close'].pct_change().iloc[-1] * 100
            st.metric("Tendance", "🟢 Haussière" if change > 0 else "🔴 Baissière")
        with cols[4]:
            if trends_score is not None:
                st.metric("Google Trends (7j)", f"{trends_score}/100")
            else:
                st.metric("Google Trends (7j)", "N/A")
                st.caption("⚠️ Google Trends ne répond pas ou limite les requêtes. Réessaie dans 1h.")
        
        if trends_series is not None:
            st.markdown("#### Intérêt Google Trends pour 'Bitcoin' (7 derniers jours)")
            st.line_chart(trends_series)
        
    def display_sentiment_analysis(self):
        # Récupération des sentiments récents
        sentiments = self.sentiment_analyzer.sentiment_collection.find().sort([("timestamp", -1)]).limit(100)
        df_sentiment = pd.DataFrame(list(sentiments))
        if df_sentiment is None or df_sentiment.empty or 'timestamp' not in df_sentiment.columns or 'composite_sentiment' not in df_sentiment.columns:
            st.warning("Aucune donnée de sentiment disponible.")
            return
        
        # Création du graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sentiment['timestamp'],
            y=df_sentiment['composite_sentiment'],
            name="Sentiment Composite"
        ))
        st.plotly_chart(fig)
    
    def display_historical_data(self):
        # Affichage des données historiques
        historical_data = self.data_collector.historical_collection.find()
        df_historical = pd.DataFrame(list(historical_data))
        if df_historical is None or df_historical.empty:
            st.warning("Aucune donnée historique disponible.")
            return
        st.dataframe(df_historical)

def get_google_trends_score_cached(keyword="Bitcoin", timeframe="now 7-d", geo=""):
    cache_file = "trends_cache.pkl"
    now = int(time.time() // 3600)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        if (
            cache.get("hour") == now and
            cache.get("keyword") == keyword and
            cache.get("timeframe") == timeframe and
            cache.get("geo") == geo
        ):
            return cache["score"], cache["series"]
    # Sinon, nouvelle requête
    try:
        pytrends = TrendReq(hl='fr-FR', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
        data = pytrends.interest_over_time()
        if not data.empty:
            score = int(data[keyword].mean())
            series = data[keyword]
        else:
            score, series = None, None
    except Exception as e:
        st.warning(f"Erreur Google Trends : {e}")
        score, series = None, None
    with open(cache_file, "wb") as f:
        pickle.dump({
            "hour": now,
            "score": score,
            "series": series,
            "keyword": keyword,
            "timeframe": timeframe,
            "geo": geo
        }, f)
    return score, series

if __name__ == "__main__":
    dashboard = BitcoinDashboard()
    dashboard.run()