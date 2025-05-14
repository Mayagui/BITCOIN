import requests
from datetime import datetime
import pandas as pd
from typing import Dict, Any
import logging

class MarketSentimentAnalyzer:
    def __init__(self):
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.logger = logging.getLogger(__name__)

    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """Obtient l'indice Fear & Greed"""
        try:
            response = requests.get(self.fear_greed_url)
            data = response.json()
            return {
                'value': float(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification'],
                'timestamp': data['data'][0]['timestamp']
            }
        except Exception as e:
            self.logger.error(f"Erreur Fear & Greed Index: {e}")
            return {
                'value': 0,
                'classification': 'neutral',
                'timestamp': datetime.now().timestamp()
            }

    async def get_market_metrics(self) -> Dict[str, Any]:
        """Obtient les métriques de marché de CoinGecko"""
        try:
            response = requests.get(
                f"{self.coingecko_url}/coins/bitcoin",
                params={
                    'localization': 'false',
                    'tickers': 'false',
                    'community_data': 'true',
                    'developer_data': 'false'
                }
            )
            data = response.json()
            
            return {
                'price_change_24h': data['market_data']['price_change_percentage_24h'],
                'market_cap_change_24h': data['market_data']['market_cap_change_percentage_24h'],
                'reddit_subscribers': data['community_data'].get('reddit_subscribers', 0),
                'reddit_active_accounts': data['community_data'].get('reddit_accounts_active_48h', 0)
            }
        except Exception as e:
            self.logger.error(f"Erreur CoinGecko: {e}")
            return {
                'price_change_24h': 0,
                'market_cap_change_24h': 0,
                'reddit_subscribers': 0,
                'reddit_active_accounts': 0
            }

    async def aggregate_sentiments(self) -> Dict[str, Any]:
        """Agrège toutes les données de sentiment"""
        fear_greed = await self.get_fear_greed_index()
        market_metrics = await self.get_market_metrics()
        
        timestamp = datetime.now()
        
        return {
            'timestamp': timestamp,
            'fear_greed': fear_greed,
            'market_metrics': market_metrics,
            'composite_sentiment': float(fear_greed['value']) / 100  # Normalise à 0-1
        }