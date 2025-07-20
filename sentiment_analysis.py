from transformers import pipeline
import requests
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Pipeline global com correção do deprecation warning
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", 
        revision="714eb0f",
        top_k=None  # CORRIGIDO: usar top_k=None em vez de return_all_scores=True
    )
except Exception as e:
    print(f"Erro ao carregar pipeline de sentimento: {e}")
    sentiment_pipeline = None

def fetch_news(symbol='BTC'):
    try:
        if not NEWSAPI_KEY:
            print("Chave NewsAPI não configurada no .env.")
            return []
        
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': f"{symbol} OR bitcoin OR crypto",
            'apiKey': NEWSAPI_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        articles = response.json().get('articles', [])
        texts = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            
            combined_text = f"{title}. {description}".strip()
            if len(combined_text) > 10:
                texts.append(combined_text)
        
        return texts[:10]
        
    except Exception as e:
        print(f"Erro ao buscar notícias: {e}")
        return []

def sentiment_predict(symbol='BTC', min_confidence=0.6):
    try:
        if not sentiment_pipeline:
            print("Pipeline de sentimento não disponível")
            return False
        
        texts = fetch_news(symbol)
        if not texts:
            return False
        
        positive_scores = []
        
        for text in texts:
            try:
                text = text[:500]
                
                # Com top_k=None, retorna todos os scores
                result = sentiment_pipeline(text)
                
                # Procurar score POSITIVE
                for score_dict in result:
                    if score_dict['label'] == 'POSITIVE':
                        positive_scores.append(score_dict['score'])
                        break
                        
            except Exception as e:
                print(f"Erro ao processar texto: {e}")
                continue
        
        if not positive_scores:
            return False
        
        mean_score = np.mean(positive_scores)
        print(f"Sentimento médio: {mean_score:.3f} (min: {min_confidence})")
        
        return mean_score > min_confidence
        
    except Exception as e:
        print(f"Erro em sentiment_predict: {e}")
        return False
