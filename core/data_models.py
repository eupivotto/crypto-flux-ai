"""
Modelos de dados para o sistema de trading
"""
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict, Any
import random
import time

@dataclass
class Position:
    """Modelo de dados para uma posição de trading"""
    symbol: str
    entry_price: float
    amount: float
    mode: str = 'Simulado'
    id: str = None
    entry_time: datetime = None
    status: str = 'OPEN'
    take_profit_price: float = None
    stop_loss_price: float = None
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl_usd: float = 0.0
    pnl_brl: float = 0.0
    exit_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"POS_{int(time.time())}_{random.randint(1000, 9999)}"
        
        if self.entry_time is None:
            self.entry_time = datetime.now()
        
        if self.take_profit_price is None:
            from config.settings import config
            self.take_profit_price = self.entry_price * (1 + config.TAKE_PROFIT_PERCENT)
        
        if self.stop_loss_price is None:
            from config.settings import config
            self.stop_loss_price = self.entry_price * (1 - config.STOP_LOSS_PERCENT)
    
    def to_dict(self):
        """Converte para dicionário para serialização"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            data['exit_time'] = self.exit_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Cria instância a partir de dicionário"""
        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if data.get('exit_time'):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return cls(**data)

@dataclass
class TradeSignal:
    """Sinal de trading gerado pelas análises"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    timestamp: datetime
    technical_score: float
    volume_score: float
    momentum_score: float
    risk_score: float
    # NOVOS CAMPOS ADICIONADOS para compatibilidade com SignalGenerator atualizado
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validações pós-inicialização"""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence deve estar entre 0.0 e 1.0")
        
        if self.action not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError("Action deve ser 'BUY', 'SELL' ou 'HOLD'")
    
    def to_dict(self):
        """Converte para dicionário para serialização"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

# Alias para compatibilidade com código que pode usar o nome antigo
TradingSignal = TradeSignal
