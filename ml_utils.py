import pandas as pd
import joblib

def load_ml_model(path):
    return joblib.load(path)

def predict_next_day_price(model, last_row):
    features = ["open", "high", "low", "close", "volume"]
    X_pred = last_row[features]
    pred = model.predict(X_pred)[0]
    proba = model.predict_proba(X_pred)[0][1]
    return pred, proba