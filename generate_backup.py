import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def generate_backup(max_retries=3):
    print("Téléchargement des données Bitcoin depuis Yahoo Finance...")
    
    for attempt in range(max_retries):
        try:
            # Télécharger 5 ans d'historique
            df = yf.download("BTC-USD", period="5y", interval="1d")
            
            # Vérifier si les données sont valides
            if df.empty:
                raise Exception("Données vides reçues de Yahoo Finance")
                
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                raise Exception("Colonnes manquantes dans les données")
            
            # Sauvegarder dans un CSV
            df.reset_index().to_csv("bitcoin_backup.csv", index=False)
            print(f"✅ Fichier bitcoin_backup.csv créé avec succès!")
            print(f"   - Période: {df.index[0].date()} à {df.index[-1].date()}")
            print(f"   - Nombre de lignes: {len(df)}")
            return True
            
        except Exception as e:
            print(f"Tentative {attempt + 1}/{max_retries} échouée: {str(e)}")
            if attempt < max_retries - 1:
                print("Nouvelle tentative dans 5 secondes...")
                time.sleep(5)
            else:
                print("❌ Échec après toutes les tentatives")
                return False

if __name__ == "__main__":
    generate_backup() 