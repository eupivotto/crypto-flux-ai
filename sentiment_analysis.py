from transformers import pipeline
import requests
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Pipeline de análise de sentimento (usa modelo pré-treinado)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")


def fetch_news(symbol='BTC'):
    if not NEWSAPI_KEY:
        raise ValueError("Chave NewsAPI não configurada no .env.")
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    response.raise_for_status()  # Erro se falhar
    articles = response.json().get('articles', [])
    texts = [article.get('description', '') for article in articles if article.get('description')]
    return texts

def sentiment_predict(symbol='BTC', min_confidence=0.6):
    texts = fetch_news(symbol)
    if not texts:
        return False  # Default se não houver dados
    scores = []
    for text in texts:
        result = sentiment_pipeline(text)[0]
        if result['label'] == 'POSITIVE':
            scores.append(result['score'])
    mean_score = np.mean(scores) if scores else 0
    return mean_score > min_confidence  # True se sentimento positivo alto
