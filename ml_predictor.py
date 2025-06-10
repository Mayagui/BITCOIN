from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import ta
import streamlit as st
import plotly.graph_objects as go
import os
import pickle

class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.features = [
            'RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'SMA_20', 'SMA_50', 'Volume_SMA'
        ]

    def add_indicators(self, df):
        if df.empty:
            return df
        try:
            df['RSI'] = ta.momentum.rsi(df['close'])
            df['MACD'] = ta.trend.macd_diff(df['close'])
            df['BB_UPPER'] = ta.volatility.bollinger_hband(df['close'])
            df['BB_LOWER'] = ta.volatility.bollinger_lband(df['close'])
            df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df = df.fillna(0)
            return df
        except Exception as e:
            st.error(f"Erreur lors du calcul des indicateurs : {str(e)}")
            return df

    def prepare_data(self, df: pd.DataFrame, window: int = 24):
        try:
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Features manquantes: {missing_features}")
            df = df[self.features + ['close']]
            feature_data = []
            targets = []
            for i in range(len(df) - window):
                window_data = []
                for feature in self.features:
                    window_data.extend(df[feature].iloc[i:i+window].values)
                feature_data.append(window_data)
                targets.append(df['close'].iloc[i+window])
            if not feature_data:
                raise ValueError("Pas assez de données pour créer des features")
            X = np.array(feature_data)
            X = self.imputer.fit_transform(X)
            X = self.scaler.fit_transform(X)
            y = np.array(targets)
            return X, y
        except Exception as e:
            raise ValueError(f"Erreur dans la préparation des données : {str(e)}")

    def train(self, X, y, model_path="ml_predictor.pkl"):
        try:
            self.model.fit(X, y)
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
        except Exception as e:
            raise ValueError(f"Erreur lors de l'entraînement : {str(e)}")

    def load_model(self, model_path="ml_predictor.pkl"):
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        return False

    def predict(self, X):
        try:
            return self.model.predict(X)[0]
        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction : {str(e)}")

    def get_model_score(self, X, y):
        try:
            return self.model.score(X, y)
        except Exception as e:
            raise ValueError(f"Erreur lors du calcul du score : {str(e)}")

    def display_predictions(self, data):
        try:
            st.subheader("Prédictions")
            predictor = PricePredictor()
            X, y = predictor.prepare_data(data)
            if len(X) > 0 and len(y) > 0:
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                if not predictor.load_model():
                    predictor.train(X_train, y_train)
                score = predictor.get_model_score(X_test, y_test)
                last_window = X[-1].reshape(1, -1)
                prediction = predictor.predict(last_window)
                col1, col2 = st.columns(2)
                with col1:
                    current_price = data['close'].iloc[-1]
                    price_change = ((prediction - current_price) / current_price) * 100
                    st.metric("Prix prédit (24h)", f"${prediction:,.2f}", f"{price_change:+.2f}%")
                with col2:
                    st.metric("Précision du modèle", f"{score*100:.1f}%")
                with st.expander("Détails du modèle"):
                    st.write("Caractéristiques utilisées :", predictor.features)
                    st.write("Taille des données d'entraînement :", len(X_train))
                    st.write("Taille des données de test :", len(X_test))
                y_pred = [predictor.predict(x.reshape(1, -1)) for x in X_test]
                st.write("### Évolution des prédictions vs Réel")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Réel'))
                fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Prédit'))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur de prédiction: {str(e)}")