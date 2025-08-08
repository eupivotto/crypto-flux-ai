from venv import logger
from config.exchange_config import exchange


def get_testnet_balance():
    try:
        # Retorna o saldo da moeda "USDT" (geralmente usada como quote)
        balance =  exchange.fetch_balance()
        usdt_balance = balance['total'].get('USDT', 0.0)
        return usdt_balance
    except Exception as e:
        logger.error(f"Erro ao obter saldo Testnet: {e}")
        return None
