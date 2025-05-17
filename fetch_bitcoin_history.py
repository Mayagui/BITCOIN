import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

def fetch_historical_data(symbol="BTC-USD", interval="1h"):
    if interval == "1h":
        period = "2y"
    elif interval == "1d":
        period = "5y"
    else:
        period = "1mo"
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty:
            logging.warning("Aucune donnée récupérée pour %s.", symbol)
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données : {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = fetch_historical_data()
    if not df.empty:
        df.to_csv("bitcoin_5y.csv", index=False)
        print(f"{len(df)} lignes sauvegardées dans bitcoin_5y.csv")
        print(df.shape)
        print(df.head())
    else:
        print("Aucune donnée sauvegardée.")
