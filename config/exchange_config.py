"""
Configuração da exchange Binance
"""
import ccxt
import os
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger(__name__)

def setup_exchange():
    """Configura e retorna instância da exchange Binance"""
    load_dotenv()
    
    try:
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Chaves da API não encontradas no .env")
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'spot'}
        })
        
        exchange.set_sandbox_mode(True)  # Testnet
        
        # Teste de conectividade
        exchange.load_markets()
        logger.info("Exchange Binance configurada com sucesso")
        
        return exchange
        
    except Exception as e:
        logger.error(f"Erro na configuração da exchange: {e}")
        raise

# Instância global da exchange
exchange = setup_exchange()
