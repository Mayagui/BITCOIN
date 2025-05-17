import time
import logging
from data_collector import BitcoinDataCollector
from sentiment_analyzer import EnhancedSentimentAnalyzer

# Initialisation du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class AutomationManager:
    def __init__(self):
        self.data_collector = BitcoinDataCollector()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.logger = logging.getLogger(__name__)

    def run_sentiment_analysis(self):
        """Exécute l'analyse des sentiments"""
        try:
            self.logger.info("Début de l'analyse des sentiments.")
            self.sentiment_analyzer.aggregate_sentiments()
            self.logger.info("Analyse des sentiments terminée.")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des sentiments : {e}")

    def update_historical_data(self):
        """Met à jour les données historiques"""
        try:
            self.logger.info("Mise à jour des données historiques...")
            data = self.data_collector.fetch_historical_data(years=5)
            self.data_collector.store_historical_data(data)
            self.logger.info("Données historiques mises à jour.")
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des données historiques : {e}")

    def main(self):
        """Boucle principale d'automatisation"""
        self.logger.info("Démarrage de l'automatisation Bitcoin.")
        try:
            while True:
                try:
                    self.logger.info("Début du stream temps réel.")
                    self.data_collector.realtime_data_stream()
                    self.logger.info("Stream temps réel terminé.")

                    self.run_sentiment_analysis()
                    self.update_historical_data()
                except Exception as e:
                    self.logger.error(f"Erreur dans la boucle principale : {e}")
                time.sleep(60)  # Mise à jour toutes les minutes
        except KeyboardInterrupt:
            self.logger.info("Arrêt manuel de l'automatisation (KeyboardInterrupt).")

if __name__ == "__main__":
    automation = AutomationManager()
    automation.main()