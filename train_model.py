import pandas as pd
import ta
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Charger les données
df = pd.read_csv("bitcoin_5y.csv", parse_dates=["timestamp"])

# Vérification des colonnes
required_cols = {'close', 'volume'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Colonnes manquantes dans le CSV : {required_cols - set(df.columns)}")

# Calculer les indicateurs
df['rsi'] = ta.momentum.rsi(df['close'])
df['macd'] = ta.trend.macd(df['close'])
df['macd_signal'] = ta.trend.macd_signal(df['close'])

# Créer la cible : 1 si le prix monte le lendemain, 0 sinon
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna(subset=['rsi', 'macd', 'macd_signal', 'volume', 'target'])

features = ['rsi', 'macd', 'macd_signal', 'volume']
X = df[features]
y = df['target']

# Split pour validation rapide
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validation rapide
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
logging.info(f"Score de validation (accuracy): {acc:.2%}")

# Sauvegarder le modèle
with open("ml_model.pkl", "wb") as f:
    pickle.dump(model, f)

logging.info("Modèle entraîné et sauvegardé sous ml_model.pkl")