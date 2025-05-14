import ccxt
import pandas as pd
from datetime import datetime, timedelta

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

if __name__ == "__main__":
    df = fetch_historical_data()
    df.to_csv("bitcoin_5y.csv", index=False)
    print(f"{len(df)} lignes sauvegardées dans bitcoin_5y.csv")
    