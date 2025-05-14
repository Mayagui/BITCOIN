import asyncio
import schedule
import time
from data_collector import BitcoinDataCollector
from sentiment_analyzer import EnhancedSentimentAnalyzer
import logging

class AutomationManager:
    def __init__(self):
        self.data_collector = BitcoinDataCollector()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.logger = logging.getLogger(__name__)

    async def run_sentiment_analysis(self):
        """Exécute l'analyse des sentiments"""
        try:
            await self.sentiment_analyzer.aggregate_sentiments()
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")

    async def update_historical_data(self):
        """Met à jour les données historiques"""
        try:
            data = self.data_collector.fetch_historical_data(years=5)
            self.data_collector.store_historical_data(data)
        except Exception as e:
            self.logger.error(f"Error updating historical data: {e}")

    async def main(self):
        """Fonction principale d'automatisation"""
        while True:
            await asyncio.gather(
                self.data_collector.realtime_data_stream(),
                self.run_sentiment_analysis(),
                self.update_historical_data()
            )
            await asyncio.sleep(60)  # Mise à jour toutes les minutes

if __name__ == "__main__":
    automation = AutomationManager()
    asyncio.run(automation.main()) 