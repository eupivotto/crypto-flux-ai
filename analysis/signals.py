"""
Gerador de sinais de trading
"""
import streamlit as st
from typing import Optional, Dict, List
from datetime import datetime
from analysis.market_analyzer import MarketAnalyzer
from core.data_models import TradeSignal
from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class SignalGenerator:
    """Gerador de sinais de trading baseado em múltiplas análises"""
    
    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.confidence_threshold = 0.7
        self.min_signals_required = 3
    
    def generate_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Gera sinal de trading para um símbolo"""
        try:
            # Obter análise completa do mercado
            market_analysis = self.market_analyzer.get_comprehensive_analysis(symbol)
            
            if not market_analysis:
                logger.error(f"Não foi possível obter análise de mercado para {symbol}")
                return None
            
            # Gerar sinais individuais
            technical_signal = self._generate_technical_signal(market_analysis)
            volume_signal = self._generate_volume_signal(market_analysis)
            momentum_signal = self._generate_momentum_signal(market_analysis)
            risk_signal = self._generate_risk_signal(symbol)
            
            # Combinar sinais
            final_signal = self._combine_signals(
                technical_signal, volume_signal, momentum_signal, risk_signal
            )
            
            if final_signal:
                trade_signal = TradeSignal(
                    symbol=symbol,
                    action=final_signal['action'],
                    confidence=final_signal['confidence'],
                    timestamp=datetime.now(),
                    technical_score=technical_signal['score'],
                    volume_score=volume_signal['score'],
                    momentum_score=momentum_signal['score'],
                    risk_score=risk_signal['score']
                )
                
                logger.info(f"Sinal gerado para {symbol}: {final_signal['action']} (conf: {final_signal['confidence']:.2f})")
                return trade_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinal para {symbol}: {e}")
            return None
    
    def _generate_technical_signal(self, market_analysis: Dict) -> Dict:
        """Gera sinal baseado em análise técnica"""
        try:
            trend = market_analysis.get('trend', {})
            score = 0.5  # Neutro
            
            if trend.get('trend') == 'BULLISH':
                score = 0.5 + (trend.get('strength', 0) * 0.5)
            elif trend.get('trend') == 'BEARISH':
                score = 0.5 - (trend.get('strength', 0) * 0.5)
            
            # Ajustar por confiança
            confidence = trend.get('confidence', 0.5)
            adjusted_score = score * confidence + 0.5 * (1 - confidence)
            
            return {
                'signal': 'BUY' if adjusted_score > 0.6 else 'SELL' if adjusted_score < 0.4 else 'HOLD',
                'score': adjusted_score,
                'strength': trend.get('strength', 0),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Erro no sinal técnico: {e}")
            return {'signal': 'HOLD', 'score': 0.5, 'strength': 0, 'confidence': 0}
    
    def _generate_volume_signal(self, market_analysis: Dict) -> Dict:
        """Gera sinal baseado em volume"""
        try:
            volume = market_analysis.get('volume', {})
            score = 0.5  # Neutro
            
            volume_signal = volume.get('volume_signal', 'NORMAL')
            volume_trend = volume.get('volume_trend', 'STABLE')
            
            if volume_signal == 'HIGH' and volume_trend == 'INCREASING':
                score = 0.8
            elif volume_signal == 'HIGH':
                score = 0.7
            elif volume_signal == 'LOW':
                score = 0.3
            
            return {
                'signal': 'BUY' if score > 0.6 else 'SELL' if score < 0.4 else 'HOLD',
                'score': score,
                'volume_ratio': volume.get('volume_ratio', 1.0)
            }
            
        except Exception as e:
            logger.error(f"Erro no sinal de volume: {e}")
            return {'signal': 'HOLD', 'score': 0.5, 'volume_ratio': 1.0}
    
    def _generate_momentum_signal(self, market_analysis: Dict) -> Dict:
        """Gera sinal baseado em momentum"""
        try:
            momentum = market_analysis.get('momentum', {})
            momentum_signal = momentum.get('momentum', 'NEUTRAL')
            rsi = momentum.get('rsi', 50)
            
            # Score baseado no momentum
            momentum_scores = {
                'STRONG_BUY': 0.9,
                'BUY': 0.7,
                'NEUTRAL': 0.5,
                'SELL': 0.3,
                'STRONG_SELL': 0.1
            }
            
            score = momentum_scores.get(momentum_signal, 0.5)
            
            # Ajustar por RSI
            if rsi > 70:  # Sobrecomprado
                score *= 0.8
            elif rsi < 30:  # Sobrevendido
                score = min(score + 0.2, 1.0)
            
            return {
                'signal': 'BUY' if score > 0.6 else 'SELL' if score < 0.4 else 'HOLD',
                'score': score,
                'rsi': rsi,
                'momentum': momentum_signal
            }
            
        except Exception as e:
            logger.error(f"Erro no sinal de momentum: {e}")
            return {'signal': 'HOLD', 'score': 0.5, 'rsi': 50, 'momentum': 'NEUTRAL'}
    
    def _generate_risk_signal(self, symbol: str) -> Dict:
        """Gera sinal baseado em gestão de risco"""
        try:
            # Verificar número de posições abertas
            open_positions = len(getattr(st.session_state, 'open_positions', []))
            
            if open_positions >= config.MAX_OPEN_POSITIONS:
                return {'signal': 'HOLD', 'score': 0.0, 'reason': 'max_positions'}
            
            # Verificar trades recentes
            trades_today = getattr(st.session_state, 'today_trades_count', 0)
            if trades_today >= config.MAX_TRADES_PER_DAY:
                return {'signal': 'HOLD', 'score': 0.0, 'reason': 'max_trades'}
            
            # Verificar perdas consecutivas
            consecutive_losses = getattr(st.session_state, 'consecutive_losses', 0)
            
            if consecutive_losses >= 3:
                return {'signal': 'HOLD', 'score': 0.2, 'reason': 'consecutive_losses'}
            elif consecutive_losses >= 2:
                return {'signal': 'HOLD', 'score': 0.6, 'reason': 'caution'}
            else:
                return {'signal': 'BUY', 'score': 1.0, 'reason': 'risk_ok'}
            
        except Exception as e:
            logger.error(f"Erro no sinal de risco: {e}")
            return {'signal': 'HOLD', 'score': 0.5, 'reason': 'error'}
    
    def _combine_signals(self, technical: Dict, volume: Dict, momentum: Dict, risk: Dict) -> Optional[Dict]:
        """Combina todos os sinais em decisão final"""
        try:
            # Pesos para cada tipo de sinal
            weights = {
                'technical': 0.4,
                'volume': 0.2,
                'momentum': 0.3,
                'risk': 0.1
            }
            
            # Se risco é negativo, bloquear trade
            if risk['score'] < 0.5:
                logger.info(f"Trade bloqueado por risco: {risk['reason']}")
                return None
            
            # Calcular score ponderado
            weighted_score = (
                technical['score'] * weights['technical'] +
                volume['score'] * weights['volume'] +
                momentum['score'] * weights['momentum'] +
                risk['score'] * weights['risk']
            )
            
            # Contar sinais positivos
            positive_signals = sum([
                1 for signal in [technical, volume, momentum, risk]
                if signal['signal'] == 'BUY'
            ])
            
            # Determinar ação final
            if weighted_score > 0.7 and positive_signals >= self.min_signals_required:
                action = 'BUY'
                confidence = weighted_score
            elif weighted_score < 0.3:
                action = 'SELL'
                confidence = 1 - weighted_score
            else:
                action = 'HOLD'
                confidence = 0.5
            
            # Verificar threshold mínimo
            if confidence < self.confidence_threshold and action != 'HOLD':
                logger.info(f"Confiança insuficiente: {confidence:.2f} < {self.confidence_threshold}")
                return None
            
            return {
                'action': action,
                'confidence': confidence,
                'weighted_score': weighted_score,
                'positive_signals': positive_signals,
                'signals_detail': {
                    'technical': technical,
                    'volume': volume,
                    'momentum': momentum,
                    'risk': risk
                }
            }
            
        except Exception as e:
            logger.error(f"Erro ao combinar sinais: {e}")
            return None
    
    def set_confidence_threshold(self, threshold: float):
        """Define threshold de confiança"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        logger.info(f"Threshold de confiança definido para: {self.confidence_threshold}")
    
    def set_min_signals_required(self, min_signals: int):
        """Define número mínimo de sinais positivos"""
        self.min_signals_required = max(1, min(4, min_signals))
        logger.info(f"Mínimo de sinais positivos: {self.min_signals_required}")
