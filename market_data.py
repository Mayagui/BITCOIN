import os
import pandas as pd
import streamlit as st
import ta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import io

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
def load_google_trends(df):
    dates = df.index  # Utilise exactement les mêmes dates que df
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
    features = features.fillna(method="ffill").dropna()
    target = target.iloc[:len(features)]
    if len(features) < 30:
        return None, None, None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    last_row = features.iloc[[-1]]
    pred = model.predict(scaler.transform(last_row))[0]
    proba = model.predict_proba(scaler.transform(last_row))[0][1]
    return pred, proba, model

# --- Application principale ---

def main():
    custom_css()
    st.title("📊 Analyse du Marché Bitcoin (Optimisée & Complète)")

    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="logo.svg" width="120" alt="Bitcoin Logo" />
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sélection de la période AVANT le chargement des données
    if "periode" not in st.session_state:
        st.session_state.periode = "5 ans"
    periode = st.sidebar.selectbox(
        "Période",
        ["7 jours", "1 mois", "3 mois", "6 mois", "1 an", "2 ans", "5 ans", "Tout"],
        index=["7 jours", "1 mois", "3 mois", "6 mois", "1 an", "2 ans", "5 ans", "Tout"].index(st.session_state.periode),
        key="periode"
    )
    # st.session_state.periode sera mis à jour automatiquement

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

    # Charge les données
    df = load_bitcoin_data()
    if nb_jours is not None:
        last_date = df.index.max()
        date_min = last_date - pd.Timedelta(days=nb_jours)
        df = df[df.index >= date_min]

    # Page d'accueil / Mode d'emploi
    with st.expander("ℹ️ Mode d'emploi de l'application", expanded=True):
        st.markdown("""
        **Bienvenue sur votre tableau de bord Bitcoin !**

        - Naviguez via le menu latéral pour explorer les différentes analyses.
        - **Vue Marché** : Aperçu global du marché et des volumes.
        - **Indicateurs** : Visualisez les principaux indicateurs techniques.
        - **Signaux** : Consultez les signaux d'achat/vente générés automatiquement.
        - **ML** : Testez la prédiction de tendance avec différents modèles et périodes.
        - **Google Trends** : Suivi de l'intérêt Google pour Bitcoin.
        - **Données brutes** : Accédez à toutes les données utilisées.

        _Utilisez les expandeurs et filtres pour personnaliser votre analyse !_
        """)

    onglet = st.sidebar.radio("Navigation", [
        "Vue Marché", "Indicateurs", "Signaux", "ML", "Google Trends", "Données brutes"
    ])

    df = compute_indicators(df)
    trends = load_google_trends(df)

    # Après avoir chargé trends et df, aligne les index
    df = df.copy()
    trends = trends.reindex(df.index).fillna(method="ffill")
    df["google_trends"] = trends["bitcoin"]

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
            st.caption("RSI (Relative Strength Index) : Indique si le marché est suracheté (>70) ou survendu (<30).")
            st.line_chart(df["RSI"].dropna())

        with st.expander("📈 MACD & Signal"):
            st.caption("MACD : Indicateur de momentum basé sur deux moyennes mobiles. Croisement = signal d'achat/vente.")
            st.line_chart(df[["MACD", "MACD_Signal"]].dropna())

        with st.expander("📈 Bandes de Bollinger"):
            st.caption("Bandes de Bollinger : Mesurent la volatilité autour d'une moyenne mobile.")
            st.line_chart(df[["close", "BB_Upper", "BB_Lower"]].dropna())

        with st.expander("📈 ADX"):
            st.caption("ADX : Mesure la force d'une tendance (au-dessus de 25 = tendance forte).")
            st.line_chart(df["ADX"].dropna())

        with st.expander("📈 OBV"):
            st.caption("OBV (On Balance Volume) : Indicateur de flux de volume pour détecter les mouvements de fonds.")
            st.line_chart(df["OBV"].dropna())

        st.download_button(
            label="📥 Télécharger ces données (CSV)",
            data=df.tail(30).to_csv().encode('utf-8'),
            file_name="indicateurs_bitcoin.csv",
            mime="text/csv"
        )

    elif onglet == "Signaux":
        st.subheader("Signaux personnalisés & scoring")
        df_signals = compute_signals(df)
        st.dataframe(df_signals[["RSI", "MACD", "signal_rsi", "signal_macd", "signal_stochrsi", "signal_williams", "score"]].tail(30))

        # Alerte si score max sur la dernière ligne
        if df_signals["score"].iloc[-1] >= 3:
            st.warning("⚡ Signal fort détecté sur la dernière journée !")

        st.download_button(
            label="📥 Télécharger ces signaux (CSV)",
            data=df_signals[["RSI", "MACD", "signal_rsi", "signal_macd", "signal_stochrsi", "signal_williams", "score"]].tail(30).to_csv().encode('utf-8'),
            file_name="signaux_bitcoin.csv",
            mime="text/csv"
        )

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

        modele = st.selectbox("Modèle ML", ["RandomForest", "LogisticRegression", "SVM"], index=0)

        if modele == "RandomForest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif modele == "LogisticRegression":
            model = LogisticRegression()
        elif modele == "SVM":
            model = SVC(probability=True)

        # Historique des prédictions ML
        features = df_ml[[
            "open", "high", "low", "close", "volume",
            "RSI", "MACD", "MACD_Signal"
        ]].dropna()
        st.write(f"Nombre de lignes utilisables pour le ML : {len(features)}")
        st.write("Proportion de NaN par colonne :")
        st.write(df_ml[[
            "open", "high", "low", "close", "volume",
            "RSI", "MACD", "MACD_Signal", "Stoch_RSI", "Williams_%R", "CCI", "ATR", "BB_Upper", "BB_Lower", "ADX", "OBV",
            "google_trends"
        ]].isna().mean())
        horizon = st.selectbox("Horizon de prédiction", [1, 2, 3, 7], index=0, format_func=lambda x: f"{x} jour{'s' if x > 1 else ''}")
        target = (df_ml["close"].shift(-horizon) > df_ml["close"]).astype(int).dropna()
        features = features.iloc[:-horizon]
        target = target.iloc[:len(features)]
        if len(features) >= 30:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            df_pred = features.copy()
            df_pred["ML_Prediction"] = model.predict(X_scaled)
            df_pred["True_Trend"] = target.values

            # Prédiction pour demain
            last_row = features.iloc[[-1]]
            pred_tomorrow = model.predict(scaler.transform(last_row))[0]
            proba_tomorrow = model.predict_proba(scaler.transform(last_row))[0][1]
            st.info(f"**Prédiction pour demain :** {'Hausse' if pred_tomorrow == 1 else 'Baisse'} (Confiance : {proba_tomorrow:.2%})")
            st.info(
                """
                **Comment fonctionne la prédiction de demain ?**

                Le modèle analyse les données du dernier jour (prix, volume, indicateurs techniques comme RSI, MACD, etc.)
                et compare cette configuration à toutes celles qu'il a vues dans l'historique. 
                Il prédit ensuite si le prix de clôture du jour suivant sera plus haut (hausse) ou plus bas (baisse).
                La confiance affichée correspond à la probabilité estimée par le modèle.
                > Le modèle ne voit pas le futur : il se base uniquement sur les patterns passés.
                """
            )
            st.write("**Valeurs utilisées pour la prédiction de demain :**")
            st.write(last_row)

            # Prédiction multi-horizon
            st.subheader("Prédiction multi-horizon")
            for h in [1, 2, 3, 7]:
                target_h = (df_ml["close"].shift(-h) > df_ml["close"]).astype(int).dropna()
                features_h = features.iloc[:-h]
                target_h = target_h.iloc[:len(features_h)]
                if len(features_h) >= 30:
                    scaler_h = StandardScaler()
                    X_scaled_h = scaler_h.fit_transform(features_h)
                    model.fit(X_scaled_h, target_h)
                    last_row_h = features_h.iloc[[-1]]
                    pred_h = model.predict(scaler_h.transform(last_row_h))[0]
                    proba_h = model.predict_proba(scaler_h.transform(last_row_h))[0][1]
                    st.write(f"**Dans {h} jour(s)** : {'Hausse' if pred_h == 1 else 'Baisse'} (Confiance : {proba_h:.2%})")

            # Affichage historique
            st.line_chart(df_pred[["ML_Prediction", "True_Trend"]])
            st.dataframe(df_pred.tail(30))

            # Précision du modèle sur la période
            st.success(f"**Précision du modèle sur la période : {acc:.2%}**")

            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            total_preds = cm.sum()
            good_preds = np.trace(cm)
            accuracy = good_preds / total_preds
            st.write(f"Nombre total de prédictions : {total_preds}")
            st.write(f"Taux de réussite (accuracy) : {accuracy:.2%}")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Baisse", "Hausse"])
            disp.plot(ax=ax)
            st.pyplot(fig)
        else:
            st.info("Pas assez de données pour afficher l'historique des prédictions ML.")

    elif onglet == "Google Trends":
        st.subheader("Google Trends (exemple)")
        trends = load_google_trends(df)
        st.line_chart(trends)

    elif onglet == "Données brutes":
        st.dataframe(df)
        st.download_button(
            label="📥 Télécharger toutes les données (CSV)",
            data=df.to_csv().encode('utf-8'),
            file_name="donnees_brutes_bitcoin.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()