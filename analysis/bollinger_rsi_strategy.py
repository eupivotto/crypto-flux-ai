"""
Estratégia Bollinger Bands + RSI + Volatilidade
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BollingerRSISignal:
    """Sinal da estratégia Bollinger + RSI"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price: float
    rsi_value: float
    bb_position: float  # posição do preço nas bandas (-1 a 1)
    volatility: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    entry_reason: str

class BollingerRSIStrategy:
    """Estratégia combinada de Bollinger Bands + RSI + Volatilidade"""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, 
                 rsi_period: int = 14, min_volatility: float = 0.005):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.min_volatility = min_volatility
        self.logger = get_logger("BollingerRSIStrategy")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calcula Bollinger Bands, RSI e Volatilidade"""
        try:
            # Bollinger Bands
            bb_middle = data['close'].rolling(window=self.bb_period).mean()
            bb_std = data['close'].rolling(window=self.bb_period).std()
            bb_upper = bb_middle + (bb_std * self.bb_std)
            bb_lower = bb_middle - (bb_std * self.bb_std)
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Volatilidade (largura das bandas normalizada)
            volatility = (bb_upper - bb_lower) / bb_middle
            
            # Posição do preço nas bandas (-1 = na banda inferior, 1 = na banda superior)
            bb_position = (data['close'] - bb_lower) / (bb_upper - bb_lower) - 0.5
            bb_position = bb_position * 2  # Normaliza para -1 a 1
            
            return {
                'bb_upper': bb_upper.iloc[-1],
                'bb_middle': bb_middle.iloc[-1],
                'bb_lower': bb_lower.iloc[-1],
                'rsi': rsi.iloc[-1],
                'volatility': volatility.iloc[-1],
                'bb_position': bb_position.iloc[-1],
                'current_price': data['close'].iloc[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao calcular indicadores: {e}")
            return None
    
    def generate_signal(self, data: pd.DataFrame) -> Optional[BollingerRSISignal]:
        """Gera sinal baseado na estratégia Bollinger + RSI"""
        try:
            if len(data) < max(self.bb_period, self.rsi_period) + 5:
                return None
                
            indicators = self.calculate_indicators(data)
            if not indicators:
                return None
            
            price = indicators['current_price']
            rsi = indicators['rsi']
            volatility = indicators['volatility']
            bb_position = indicators['bb_position']
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            bb_middle = indicators['bb_middle']
            
            # Verifica volatilidade mínima
            if volatility < self.min_volatility:
                return BollingerRSISignal(
                    action='HOLD',
                    confidence=0.0,
                    price=price,
                    rsi_value=rsi,
                    bb_position=bb_position,
                    volatility=volatility,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    bb_middle=bb_middle,
                    entry_reason="Volatilidade insuficiente"
                )
            
            # SINAL DE COMPRA
            # Preço rompe banda superior + RSI > 70 (momentum forte)
            if (price > bb_upper and rsi > 70 and volatility > self.min_volatility):
                confidence = min(0.95, 
                    0.6 + (rsi - 70) / 100 + volatility * 2)  # Aumenta confiança com RSI alto e volatilidade
                
                return BollingerRSISignal(
                    action='BUY',
                    confidence=confidence,
                    price=price,
                    rsi_value=rsi,
                    bb_position=bb_position,
                    volatility=volatility,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    bb_middle=bb_middle,
                    entry_reason=f"Breakout banda superior: RSI={rsi:.1f}, Vol={volatility:.3f}"
                )
            
            # SINAL DE COMPRA ALTERNATIVO
            # Preço toca banda inferior + RSI < 30 (sobrevenda com possível reversão)
            elif (price <= bb_lower * 1.001 and rsi < 30 and volatility > self.min_volatility):
                confidence = min(0.85,
                    0.5 + (30 - rsi) / 100 + volatility * 1.5)
                
                return BollingerRSISignal(
                    action='BUY',
                    confidence=confidence,
                    price=price,
                    rsi_value=rsi,
                    bb_position=bb_position,
                    volatility=volatility,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    bb_middle=bb_middle,
                    entry_reason=f"Oversold bounce: RSI={rsi:.1f}, Vol={volatility:.3f}"
                )
            
            # HOLD para outros casos
            return BollingerRSISignal(
                action='HOLD',
                confidence=0.3,
                price=price,
                rsi_value=rsi,
                bb_position=bb_position,
                volatility=volatility,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_middle=bb_middle,
                entry_reason=f"Aguardando setup: RSI={rsi:.1f}, BB_pos={bb_position:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal: {e}")
            return None
