"""
Configurações globais do AI Trading Bot
"""
import os
from dataclasses import dataclass, field

@dataclass
class TradingConfig:
    # Configurações de Trading para Scalping
    DAILY_TARGET: float = 50.0
    DAILY_TARGET_USD: float = 9.09
    TRADE_AMOUNT: float = 0.001
    SCALPER_MODE: bool = True
    MIN_PROFIT_TARGET: float = 0.005
    MAX_TRADES_PER_DAY: int = 200
    MIN_INTERVAL_BETWEEN_TRADES: int = 30  
    STOP_LOSS_PERCENT: float = 0.0008      # 0.08% (mais conservador)
    TAKE_PROFIT_PERCENT: float = 0.006     # 0.6% (relação 1:7.5)
    USD_TO_BRL: float = 5.5
    
    # Configurações do Sistema de Posições para Scalping
    POSITION_MONITORING_INTERVAL: int = 1  
    ORDER_TIMEOUT: int = 300
    MIN_PROFIT_TO_CLOSE: float = 0.005     
    MAX_OPEN_POSITIONS: int = 6            
    AUTO_SELL_ENABLED: bool = True
    
    # Moedas disponíveis - CORRIGIDO: inicialização direta
    AVAILABLE_COINS: list = field(default_factory=lambda: [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
        'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT',
        'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'ATOM/USDT', 'FIL/USDT'
    ])

# Instância global das configurações
config = TradingConfig()
