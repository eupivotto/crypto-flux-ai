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
from analysis.bollinger_rsi_strategy import BollingerRSIStrategy

logger = get_logger(__name__)

class SignalGenerator:
    """Gerador de sinais de trading baseado em múltiplas análises"""

    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.confidence_threshold = 0.5
        self.min_signals_required = 2
        
        # Parâmetro para ativar/desativar o modo scalper
        self.scalper_mode = getattr(config, "SCALPER_MODE", True)
        
        # Nova estratégia Bollinger+RSI
        self.bollinger_rsi = BollingerRSIStrategy(
            bb_period=st.session_state.get('bb_period', 20),
            bb_std=st.session_state.get('bb_std', 2.0),
            rsi_period=st.session_state.get('rsi_period', 14),
            min_volatility=st.session_state.get('min_volatility', 0.005)
        )

    def generate_signal(self, symbol: str) -> Optional[TradeSignal]:
      """Gera sinal de trading para um símbolo usando estratégia selecionada"""
      try:
        # Obter análise completa do mercado
        market_analysis = self.market_analyzer.get_comprehensive_analysis(symbol)
        if not market_analysis:
            logger.error(f"Não foi possível obter análise de mercado para {symbol}")
            return None

        # Pega dataframe cru para análises
        df = self.market_analyzer.get_market_data(symbol)
        if df is None or len(df) < 50:
            logger.warning(f"Dados insuficientes para {symbol}")
            return None

        # ⭐ SEMPRE CALCULAR AS 4 IAs PARA O DASHBOARD (INDEPENDENTE DA ESTRATÉGIA)
        technical_signal = self._generate_technical_signal(market_analysis)
        volume_signal = self._generate_volume_signal(market_analysis)
        momentum_signal = self._generate_momentum_signal(market_analysis)
        risk_signal = self._generate_risk_signal(symbol)

        # Atualizar predictions no session_state para o dashboard
        st.session_state.predictions = [
            technical_signal['score'] > 0.6,
            volume_signal['score'] > 0.6,
            momentum_signal['score'] > 0.6,
            risk_signal['score'] > 0.6
        ]
        
        # Determinar decisão geral das IAs
        positive_count = sum(st.session_state.predictions)
        st.session_state.decision = positive_count >= 3

        # Verificar qual estratégia usar
        strategy_mode = st.session_state.get('strategy_mode', 'Scalper (Volume/Candle)')
        
        # 🎯 ESTRATÉGIA: Bollinger+RSI+Volatilidade
        if strategy_mode in ['Bollinger+RSI+Vol', 'Ambas']:
            bb_signal = self._generate_bollinger_rsi_signal(df)
            if bb_signal and bb_signal['signal'] == 'BUY':
                logger.info(f"🎯 Bollinger+RSI signal => {bb_signal['signal']} (conf: {bb_signal['confidence']:.2f}) - {bb_signal['reason']}")
                return TradeSignal(
                    symbol=symbol,
                    action=bb_signal['signal'],
                    confidence=bb_signal['confidence'],
                    timestamp=datetime.now(),
                    technical_score=technical_signal['score'],  # Usar scores das IAs reais
                    volume_score=volume_signal['score'],
                    momentum_score=momentum_signal['score'],
                    risk_score=risk_signal['score'],
                    additional_data={
                        'strategy': 'BollingerRSI',
                        'rsi': bb_signal['rsi'],
                        'volatility': bb_signal['volatility'],
                        'bb_position': bb_signal['bb_position'],
                        'entry_reason': bb_signal['reason'],
                        'bb_upper': bb_signal['bb_upper'],
                        'bb_lower': bb_signal['bb_lower'],
                        'bb_middle': bb_signal['bb_middle']
                    }
                )

        # 💡 ESTRATÉGIA: Scalper
        if strategy_mode in ['Scalper (Volume/Candle)', 'Ambas'] and self.scalper_mode:
            scalper_signal = self._generate_scalper_signal(df)
            if scalper_signal['signal'] in ['BUY', 'SELL']:
                logger.info(f"💡 Scalper signal ativo => {scalper_signal['signal']} (score: {scalper_signal['score']:.2f})")
                return TradeSignal(
                    symbol=symbol,
                    action=scalper_signal['signal'],
                    confidence=scalper_signal['score'],
                    timestamp=datetime.now(),
                    technical_score=technical_signal['score'],  # Usar scores das IAs reais
                    volume_score=volume_signal['score'],
                    momentum_score=momentum_signal['score'],
                    risk_score=risk_signal['score'],
                    additional_data={'strategy': 'Scalper'}
                )

        # 📊 ESTRATÉGIA: Convencional (combinada)
        if strategy_mode == 'Convencional':
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
                    risk_score=risk_signal['score'],
                    additional_data={'strategy': 'Convencional'}
                )
                logger.info(f"Sinal (convencional) gerado para {symbol}: {final_signal['action']} (conf: {final_signal['confidence']:.2f})")
                return trade_signal

            # Se chegou aqui, nenhuma estratégia gerou sinal de compra
            logger.info(f"Nenhuma estratégia gerou sinal de compra para {symbol}")
            return None

      except Exception as e:
            logger.error(f"Erro ao gerar sinal para {symbol}: {e}")
            return None


      except Exception as e:
            logger.error(f"Erro ao gerar sinal para {symbol}: {e}")
            return None

    def _generate_bollinger_rsi_signal(self, df) -> Optional[Dict]:
        """
        🎯 NOVA ESTRATÉGIA: Bollinger Bands + RSI + Volatilidade
        """
        try:
            bb_signal = self.bollinger_rsi.generate_signal(df)
            if not bb_signal:
                return None
            
            # Verificações de risco adicionais
            risk_check = self._generate_risk_signal(df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'Unknown')
            if risk_check['score'] < 0.5:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': f"Bloqueado por risco: {risk_check['reason']}"
                }
            
            # Converter o sinal da estratégia para formato interno
            momentum_score = min(1.0, bb_signal.rsi_value / 50.0) if bb_signal.rsi_value < 50 else min(1.0, (100 - bb_signal.rsi_value) / 50.0 + 0.5)
            risk_score = min(1.0, bb_signal.volatility * 10)
            
            return {
                'signal': bb_signal.action,
                'confidence': bb_signal.confidence,
                'rsi': bb_signal.rsi_value,
                'volatility': bb_signal.volatility,
                'bb_position': bb_signal.bb_position,
                'bb_upper': bb_signal.bb_upper,
                'bb_lower': bb_signal.bb_lower,
                'bb_middle': bb_signal.bb_middle,
                'momentum_score': momentum_score,
                'risk_score': risk_score,
                'reason': bb_signal.entry_reason
            }
            
        except Exception as e:
            logger.error(f"Erro no sinal Bollinger+RSI: {e}")
            return None

    def _generate_scalper_signal(self, df) -> Dict:
        """
        💡 Sinal rápido para scalping: volume, região, direção do candle.
        (Sua lógica original mantida)
        """
        try:
            if df is None or len(df) < 21:
                return {'signal': 'HOLD', 'score': 0.5}

            last = df.iloc[-1]
            avg_volume = df['volume'][-10:].mean()
            vol_ok = last['volume'] > (avg_volume * 1.2)

            # Região: toque na banda inferior de Bollinger (suporte) ou superior (resistência)
            near_lower = last['close'] <= last['bb_lower'] * 1.01
            near_upper = last['close'] >= last['bb_upper'] * 0.99

            # Direção do candle
            is_bull = last['close'] > last['open']
            is_bear = last['close'] < last['open']

            # Critério de entrada rápida para micro-operação
            if vol_ok and near_lower and is_bull:
                return {'signal': 'BUY', 'score': 1.0}
            if vol_ok and near_upper and is_bear:
                return {'signal': 'SELL', 'score': 1.0}

            return {'signal': 'HOLD', 'score': 0.5}
        except Exception as e:
            logger.error(f"Erro no scalper_signal: {e}")
            return {'signal': 'HOLD', 'score': 0.5}

    def _generate_technical_signal(self, market_analysis: Dict) -> Dict:
        """Gera sinal baseado em análise técnica (Sua lógica original)"""
        try:
            trend = market_analysis.get('trend', {})
            score = 0.5
            if trend.get('trend') == 'BULLISH':
                score = 0.5 + (trend.get('strength', 0) * 0.5)
            elif trend.get('trend') == 'BEARISH':
                score = 0.5 - (trend.get('strength', 0) * 0.5)
            confidence = trend.get('confidence', 0.5)
            adjusted_score = score * confidence + 0.5 * (1 - confidence)
            return {
                'signal': 'BUY' if adjusted_score > 0.55 else 'SELL' if adjusted_score < 0.45 else 'HOLD',
                'score': adjusted_score,
                'strength': trend.get('strength', 0),
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Erro no sinal técnico: {e}")
            return {'signal': 'HOLD', 'score': 0.5, 'strength': 0, 'confidence': 0}

    def _generate_volume_signal(self, market_analysis: Dict) -> Dict:
        """Gera sinal baseado em volume (Sua lógica original)"""
        try:
            volume = market_analysis.get('volume', {})
            score = 0.5
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
        """Gera sinal baseado em momentum (Sua lógica original)"""
        try:
            momentum = market_analysis.get('momentum', {})
            momentum_signal = momentum.get('momentum', 'NEUTRAL')
            rsi = momentum.get('rsi', 50)
            momentum_scores = {
                'STRONG_BUY': 0.9,
                'BUY': 0.7,
                'NEUTRAL': 0.5,
                'SELL': 0.3,
                'STRONG_SELL': 0.1
            }
            score = momentum_scores.get(momentum_signal, 0.5)
            if rsi > 70:
                score *= 0.8
            elif rsi < 30:
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
        """Gera sinal baseado em gestão de risco (Sua lógica original + melhorias)"""
        try:
            open_positions = len(getattr(st.session_state, 'open_positions', []))
            if open_positions >= config.MAX_OPEN_POSITIONS:
                return {'signal': 'HOLD', 'score': 0.0, 'reason': 'max_positions'}
            
            trades_today = getattr(st.session_state, 'today_trades_count', 0)
            if trades_today >= config.MAX_TRADES_PER_DAY:
                return {'signal': 'HOLD', 'score': 0.0, 'reason': 'max_trades'}
            
            consecutive_losses = getattr(st.session_state, 'consecutive_losses', 0)
            if consecutive_losses >= 3:
                return {'signal': 'HOLD', 'score': 0.2, 'reason': 'consecutive_losses'}
            elif consecutive_losses >= 2:
                return {'signal': 'HOLD', 'score': 0.6, 'reason': 'caution'}
            
            # Novo: Verificar perda diária máxima
            daily_profit = getattr(st.session_state, 'daily_profit_usd', 0.0)
            max_daily_loss = getattr(config, 'MAX_DAILY_LOSS_USD', -15.0)
            if daily_profit <= max_daily_loss:
                return {'signal': 'HOLD', 'score': 0.0, 'reason': 'max_daily_loss'}
            
            return {'signal': 'BUY', 'score': 1.0, 'reason': 'risk_ok'}
        except Exception as e:
            logger.error(f"Erro no sinal de risco: {e}")
            return {'signal': 'HOLD', 'score': 0.5, 'reason': 'error'}

    def _combine_signals(self, technical: Dict, volume: Dict, momentum: Dict, risk: Dict) -> Optional[Dict]:
        """Combina todos os sinais em decisão final (Sua lógica original)"""
        try:
            weights = {
                'technical': 0.25,
                'volume': 0.3,
                'momentum': 0.35,
                'risk': 0.1
            }
            if risk['score'] < 0.5:
                logger.info(f"Trade bloqueado por risco: {risk['reason']}")
                return None
            
            weighted_score = (
                technical['score'] * weights['technical'] +
                volume['score'] * weights['volume'] +
                momentum['score'] * weights['momentum'] +
                risk['score'] * weights['risk']
            )
            
            positive_signals = sum([
                1 for signal in [technical, volume, momentum, risk]
                if signal['signal'] == 'BUY'
            ])
            
            if weighted_score > 0.7 and positive_signals >= self.min_signals_required:
                action = 'BUY'
                confidence = weighted_score
            elif weighted_score < 0.3:
                action = 'SELL'
                confidence = 1 - weighted_score
            else:
                action = 'HOLD'
                confidence = 0.5
            
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

    def update_bollinger_settings(self):
        """Atualiza configurações da estratégia Bollinger+RSI conforme session_state"""
        try:
            self.bollinger_rsi = BollingerRSIStrategy(
                bb_period=st.session_state.get('bb_period', 20),
                bb_std=st.session_state.get('bb_std', 2.0),
                rsi_period=st.session_state.get('rsi_period', 14),
                min_volatility=st.session_state.get('min_volatility', 0.005)
            )
        except Exception as e:
            logger.error(f"Erro ao atualizar configurações Bollinger: {e}")

    def set_confidence_threshold(self, threshold: float):
        """Define threshold de confiança (Sua lógica original)"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        logger.info(f"Threshold de confiança definido para: {self.confidence_threshold}")

    def set_min_signals_required(self, min_signals: int):
        """Define número mínimo de sinais positivos (Sua lógica original)"""
        self.min_signals_required = max(1, min(4, min_signals))
        logger.info(f"Mínimo de sinais positivos: {self.min_signals_required}")
