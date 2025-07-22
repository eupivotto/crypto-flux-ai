"""
Analisador de mercado para AI Trading Bot
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from config.exchange_config import exchange
from utils.logger import get_logger

logger = get_logger(__name__)

class MarketAnalyzer:
    """Analisador avançado de condições de mercado"""
    
    def __init__(self):
        self.timeframe = '1m'
        self.limit = 100
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Obtém dados OHLCV do mercado"""
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Adicionar indicadores técnicos básicos
            df = self._add_technical_indicators(df)
            
            logger.info(f"Dados de mercado obtidos para {symbol}: {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores técnicos aos dados"""
        try:
            # Médias móveis
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR (Average True Range)
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(14).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores técnicos: {e}")
            return df
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analisa tendência do mercado"""
        try:
            if len(df) < 20:
                return {'trend': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            current_price = df['close'].iloc[-1]
            sma_5 = df['sma_5'].iloc[-1]
            sma_10 = df['sma_10'].iloc[-1]
            sma_20 = df['sma_20'].iloc[-1]
            
            # Determinar tendência baseada nas médias móveis
            if current_price > sma_5 > sma_10 > sma_20:
                trend = 'BULLISH'
                strength = (current_price - sma_20) / sma_20
            elif current_price < sma_5 < sma_10 < sma_20:
                trend = 'BEARISH'
                strength = (sma_20 - current_price) / sma_20
            else:
                trend = 'NEUTRAL'
                strength = 0.0
            
            # Calcular confiança baseada na consistência
            price_above_sma5 = sum(df['close'].tail(10) > df['sma_5'].tail(10)) / 10
            confidence = abs(price_above_sma5 - 0.5) * 2
            
            return {
                'trend': trend,
                'strength': min(abs(strength), 1.0),
                'confidence': confidence,
                'current_price': current_price,
                'sma_5': sma_5,
                'sma_20': sma_20
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de tendência: {e}")
            return {'trend': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
    
    def analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analisa momentum do mercado"""
        try:
            if len(df) < 14:
                return {'momentum': 'NEUTRAL', 'rsi': 50.0, 'macd_signal': 'NEUTRAL'}
            
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            
            # Análise RSI
            if rsi > 70:
                rsi_signal = 'OVERBOUGHT'
            elif rsi < 30:
                rsi_signal = 'OVERSOLD'
            else:
                rsi_signal = 'NEUTRAL'
            
            # Análise MACD
            if macd > macd_signal and macd > 0:
                macd_signal_str = 'BULLISH'
            elif macd < macd_signal and macd < 0:
                macd_signal_str = 'BEARISH'
            else:
                macd_signal_str = 'NEUTRAL'
            
            # Momentum geral
            if rsi_signal == 'OVERSOLD' and macd_signal_str == 'BULLISH':
                momentum = 'STRONG_BUY'
            elif rsi_signal == 'OVERBOUGHT' and macd_signal_str == 'BEARISH':
                momentum = 'STRONG_SELL'
            elif rsi > 50 and macd_signal_str == 'BULLISH':
                momentum = 'BUY'
            elif rsi < 50 and macd_signal_str == 'BEARISH':
                momentum = 'SELL'
            else:
                momentum = 'NEUTRAL'
            
            return {
                'momentum': momentum,
                'rsi': rsi,
                'rsi_signal': rsi_signal,
                'macd': macd,
                'macd_signal': macd_signal_str,
                'macd_value': macd
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de momentum: {e}")
            return {'momentum': 'NEUTRAL', 'rsi': 50.0, 'macd_signal': 'NEUTRAL'}
    
    def analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analisa volume de negociação"""
        try:
            if len(df) < 10:
                return {'volume_signal': 'NEUTRAL', 'volume_trend': 'NEUTRAL'}
            
            current_volume = df['volume'].iloc[-1]
            avg_volume_10 = df['volume'].rolling(10).mean().iloc[-1]
            avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
            
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # Sinal de volume
            if volume_ratio > 1.5:
                volume_signal = 'HIGH'
            elif volume_ratio < 0.5:
                volume_signal = 'LOW'
            else:
                volume_signal = 'NORMAL'
            
            # Tendência de volume
            if avg_volume_10 > avg_volume_20 * 1.1:
                volume_trend = 'INCREASING'
            elif avg_volume_10 < avg_volume_20 * 0.9:
                volume_trend = 'DECREASING'
            else:
                volume_trend = 'STABLE'
            
            return {
                'volume_signal': volume_signal,
                'volume_trend': volume_trend,
                'current_volume': current_volume,
                'avg_volume_20': avg_volume_20,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de volume: {e}")
            return {'volume_signal': 'NEUTRAL', 'volume_trend': 'NEUTRAL'}
    
    def analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analisa volatilidade do mercado"""
        try:
            if len(df) < 20:
                return {'volatility': 'NORMAL', 'atr': 0.0}
            
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            # Volatilidade baseada no ATR
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
            
            if atr_pct > 2.0:
                volatility = 'HIGH'
            elif atr_pct < 0.5:
                volatility = 'LOW'
            else:
                volatility = 'NORMAL'
            
            # Posição nas Bollinger Bands
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            return {
                'volatility': volatility,
                'atr': atr,
                'atr_pct': atr_pct,
                'bb_position': bb_position,
                'bb_squeeze': abs(bb_upper - bb_lower) / current_price < 0.02
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de volatilidade: {e}")
            return {'volatility': 'NORMAL', 'atr': 0.0}
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict:
        """Análise completa do mercado"""
        try:
            df = self.get_market_data(symbol)
            if df is None or df.empty:
                return {}
            
            trend_analysis = self.analyze_trend(df)
            momentum_analysis = self.analyze_momentum(df)
            volume_analysis = self.analyze_volume(df)
            volatility_analysis = self.analyze_volatility(df)
            
            # Score geral (0-100)
            score = self._calculate_overall_score(
                trend_analysis, momentum_analysis, 
                volume_analysis, volatility_analysis
            )
            
            return {
                'symbol': symbol,
                'timestamp': pd.Timestamp.now().isoformat(),
                'current_price': df['close'].iloc[-1],
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'volume': volume_analysis,
                'volatility': volatility_analysis,
                'overall_score': score,
                'recommendation': self._get_recommendation(score)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise completa de {symbol}: {e}")
            return {}
    
    def _calculate_overall_score(self, trend: Dict, momentum: Dict, volume: Dict, volatility: Dict) -> float:
        """Calcula score geral de 0-100"""
        try:
            score = 50  # Base neutra
            
            # Score baseado na tendência
            if trend['trend'] == 'BULLISH':
                score += 20 * trend['strength']
            elif trend['trend'] == 'BEARISH':
                score -= 20 * trend['strength']
            
            # Score baseado no momentum
            momentum_scores = {
                'STRONG_BUY': 25, 'BUY': 15, 'NEUTRAL': 0,
                'SELL': -15, 'STRONG_SELL': -25
            }
            score += momentum_scores.get(momentum['momentum'], 0)
            
            # Score baseado no volume
            if volume['volume_signal'] == 'HIGH' and volume['volume_trend'] == 'INCREASING':
                score += 10
            elif volume['volume_signal'] == 'LOW':
                score -= 5
            
            # Ajuste por volatilidade
            if volatility['volatility'] == 'HIGH':
                score *= 0.9  # Reduz score em alta volatilidade
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Erro no cálculo do score: {e}")
            return 50
    
    def _get_recommendation(self, score: float) -> str:
        """Converte score em recomendação"""
        if score >= 75:
            return 'STRONG_BUY'
        elif score >= 60:
            return 'BUY'
        elif score >= 40:
            return 'NEUTRAL'
        elif score >= 25:
            return 'SELL'
        else:
            return 'STRONG_SELL'
