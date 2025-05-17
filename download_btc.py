# Télécharge 5 ans de données BTC-USD depuis Yahoo Finance
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    df = yf.download("BTC-USD", period="5y", interval="1h")
    logging.info("Shape du DataFrame téléchargé : %s", df.shape)
    if df.empty:
        logging.warning("Aucune donnée téléchargée. Problème de connexion, d'API ou de ticker.")
    else:
        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "timestamp",
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Open": "open",
            "Volume": "volume"
        }, inplace=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.to_csv("bitcoin_5y.csv", index=False)
        logging.info("Fichier bitcoin_5y.csv sauvegardé avec %d lignes.", len(df))
except Exception as e:
    logging.error(f"Erreur lors du téléchargement ou de la sauvegarde : {e}")