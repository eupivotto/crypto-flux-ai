import ccxt
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

ex = ccxt.binance({'apiKey': API_KEY, 'secret': API_SECRET})
ex.set_sandbox_mode(True)  # Ativa testnet
print(ex.fetch_balance())  # Deve retornar saldo fictício
