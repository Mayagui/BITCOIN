import pickle

def load_ml_model(model_path="ml_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_next_day_price(model, last_row):
    """
    Prédit la probabilité de hausse du prix pour le prochain jour.
    last_row : DataFrame d'une seule ligne avec les colonnes ['rsi', 'macd', 'macd_signal', 'volume']
    """
    features = ['rsi', 'macd', 'macd_signal', 'volume']
    for feat in features:
        if feat not in last_row.columns:
            raise ValueError(f"Colonne manquante : {feat}")
    X_pred = last_row[features].values.reshape(1, -1)
    proba = model.predict_proba(X_pred)[0][1]
    pred = model.predict(X_pred)[0]
    return pred, proba