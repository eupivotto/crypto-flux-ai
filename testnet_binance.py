import ccxt
from dotenv import load_dotenv
import os

load_dotenv()
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_API_SECRET')
})
exchange.set_sandbox_mode(True)

# Verificar todos os saldos
balance = exchange.fetch_balance()
print("=== SALDOS COMPLETOS ===")
for asset in ['USDT', 'BTC', 'ETH', 'BNB']:
    if asset in balance and balance[asset]['total'] > 0:
        print(f"Saldo {asset}: {balance[asset]['free']}")

# Calcular valor total do portfólio em USDT
ticker = exchange.fetch_ticker('BTC/USDT')
btc_price = ticker['last']
btc_balance = balance.get('BTC', {}).get('free', 0)
usdt_balance = balance.get('USDT', {}).get('free', 0)

portfolio_value = usdt_balance + (btc_balance * btc_price)
print(f"\n=== VALOR TOTAL DO PORTFÓLIO ===")
print(f"BTC: {btc_balance:.8f} (≈ ${btc_balance * btc_price:.2f})")
print(f"USDT: {usdt_balance:.2f}")
print(f"Valor Total: ${portfolio_value:.2f}")
