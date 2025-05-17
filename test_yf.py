import yfinance as yf
df = yf.download("BTC-USD", period="5y", interval="1d")
df.reset_index(inplace=True)
df.to_csv("bitcoin_5y.csv", index=False)