from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from typing import Tuple
import ta
import streamlit as st

class PricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.features = [
            'RSI',
            'MACD',
            'BB_UPPER',
            'BB_LOWER',
            'SMA_20',
            'SMA_50',
            'Volume_SMA'
        ]
        
    def prepare_data(self, df: pd.DataFrame, window: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les données pour le ML"""
        try:
            # Vérification des colonnes
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Features manquantes: {missing_features}")
            
            # Création des features
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
            y = np.array(targets)
            
            # Gestion des valeurs manquantes
            X = self.imputer.fit_transform(X)
            
            # Normalisation
            X = self.scaler.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            raise ValueError(f"Erreur dans la préparation des données : {str(e)}")
    
    def add_indicators(self, df):
        """Ajoute les indicateurs techniques"""
        if df.empty:
            return df
        
        try:
            # RSI
            df['RSI'] = ta.momentum.rsi(df['close'])
            
            # MACD
            df['MACD'] = ta.trend.macd_diff(df['close'])
            
            # Bollinger Bands
            df['BB_UPPER'] = ta.volatility.bollinger_hband(df['close'])
            df['BB_MIDDLE'] = ta.volatility.bollinger_mavg(df['close'])
            df['BB_LOWER'] = ta.volatility.bollinger_lband(df['close'])
            
            # Moyennes mobiles
            df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            # Volume SMA - Correction ici
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            
            # Vérification que toutes les colonnes sont présentes
            required_columns = ['RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'SMA_20', 'SMA_50', 'Volume_SMA']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.warning(f"Colonnes manquantes : {missing_columns}")
            
            # Remplacement des valeurs NaN par 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            st.error(f"Erreur lors du calcul des indicateurs : {str(e)}")
            return df
    
    def display_predictions(self, data):
        """Affiche les prédictions"""
        try:
            st.subheader("Prédictions")
            
            predictor = PricePredictor()
            X, y = predictor.prepare_data(data)
            
            if len(X) > 0 and len(y) > 0:
                # Séparation train/test
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Entraînement
                predictor.train(X_train, y_train)
                
                # Score du modèle
                score = predictor.get_model_score(X_test, y_test)
                
                # Prédiction
                last_window = X[-1].reshape(1, -1)
                prediction = predictor.predict(last_window)
                
                # Affichage des résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    current_price = data['close'].iloc[-1]
                    price_change = ((prediction - current_price) / current_price) * 100
                    
                    st.metric(
                        "Prix prédit (24h)",
                        f"${prediction:,.2f}",
                        f"{price_change:+.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Précision du modèle",
                        f"{score*100:.1f}%"
                    )
                
                # Informations supplémentaires
                with st.expander("Détails du modèle"):
                    st.write("Caractéristiques utilisées :", predictor.features)
                    st.write("Taille des données d'entraînement :", len(X_train))
                    st.write("Taille des données de test :", len(X_test))
                
        except Exception as e:
            st.error(f"Erreur de prédiction: {str(e)}")
    
    def verify_indicators(self, df):
        """Vérifie que tous les indicateurs sont présents"""
        required_columns = [
            'RSI', 'MACD', 'BB_UPPER', 'BB_LOWER',
            'SMA_20', 'SMA_50', 'Volume_SMA'
        ]
        
        st.write("### Vérification des indicateurs")
        for col in required_columns:
            if col in df.columns:
                st.success(f"✓ {col} présent")
            else:
                st.error(f"✗ {col} manquant")
        
        # Afficher les premières lignes des données
        st.write("### Aperçu des données")
        st.dataframe(df[required_columns].head() if all(col in df.columns for col in required_columns) else df.head())
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Entraîne le modèle"""
        try:
            self.model.fit(X, y)
        except Exception as e:
            raise ValueError(f"Erreur lors de l'entraînement : {str(e)}")

    def predict(self, X: np.ndarray) -> float:
        """Fait une prédiction"""
        try:
            return self.model.predict(X)[0]
        except Exception as e:
            raise ValueError(f"Erreur lors de la prédiction : {str(e)}")

    def get_model_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Retourne le score du modèle"""
        try:
            return self.model.score(X, y)
        except Exception as e:
            raise ValueError(f"Erreur lors du calcul du score : {str(e)}")
    
    