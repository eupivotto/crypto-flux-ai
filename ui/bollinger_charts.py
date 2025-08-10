"""
Gráficos de alerta para estratégia Bollinger Bands + RSI
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict
from analysis.market_analyzer import MarketAnalyzer
from analysis.bollinger_rsi_strategy import BollingerRSIStrategy
from utils.logger import get_logger

logger = get_logger(__name__)

class BollingerRSICharts:
    """Classe para visualização da estratégia Bollinger+RSI"""
    
    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.logger = get_logger("BollingerRSICharts")
    
    def create_bollinger_rsi_chart(self, symbol: str, period: int = 100) -> Optional[go.Figure]:
        """
        Cria gráfico completo com Candlestick, Bollinger Bands, RSI e alertas
        """
        try:
            # Obter dados de mercado - CORRIGIDO: sem parâmetro limit
            df = self.market_analyzer.get_market_data(symbol)
            if df is None or len(df) < 50:
                self.logger.error(f"Dados insuficientes para {symbol}")
                return None
            
            # Limitar aos últimos N candles conforme período solicitado
            if len(df) > period:
                df = df.tail(period).copy()
            
            # Configurações da estratégia
            bb_period = st.session_state.get('bb_period', 20)
            bb_std = st.session_state.get('bb_std', 2.0)
            rsi_period = st.session_state.get('rsi_period', 14)
            min_volatility = st.session_state.get('min_volatility', 0.005)
            
            # Calcular indicadores
            indicators = self._calculate_indicators(df, bb_period, bb_std, rsi_period)
            
            # Identificar sinais de entrada
            signals = self._identify_signals(df, indicators, min_volatility)
            
            # Criar subplots (3 rows: Candlestick+BB, RSI, Volume)
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f'{symbol} - Bandas de Bollinger + Alertas',
                    'RSI (14) - Momentum',
                    'Volume'
                ),
                row_heights=[0.6, 0.25, 0.15]
            )
            
            # Plot 1: Candlestick + Bollinger Bands
            self._add_candlestick_plot(fig, df, indicators, signals, symbol)
            
            # Plot 2: RSI
            self._add_rsi_plot(fig, df, indicators, signals, rsi_period)
            
            # Plot 3: Volume
            self._add_volume_plot(fig, df, signals)
            
            # Configurar layout
            self._configure_layout(fig, symbol)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Erro ao criar gráfico para {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame, bb_period: int, 
                            bb_std: float, rsi_period: int) -> Dict:
        """Calcula todos os indicadores necessários"""
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Volatilidade (largura das bandas normalizada)
        volatility = (bb_upper - bb_lower) / bb_middle
        
        # Posição do preço nas bandas
        bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        return {
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'rsi': rsi,
            'volatility': volatility,
            'bb_position': bb_position
        }
    
    def _identify_signals(self, df: pd.DataFrame, indicators: Dict, 
                         min_volatility: float) -> Dict:
        """Identifica sinais de compra/venda da estratégia"""
        buy_signals = []
        sell_signals = []
        
        for i in range(len(df)):
            if i < 50:  # Skip primeiros candles
                continue
                
            current_price = df.iloc[i]['close']
            rsi_val = indicators['rsi'].iloc[i]
            bb_upper = indicators['bb_upper'].iloc[i]
            bb_lower = indicators['bb_lower'].iloc[i]
            volatility = indicators['volatility'].iloc[i]
            
            # Verificar se temos dados válidos
            if pd.isna(rsi_val) or pd.isna(bb_upper) or pd.isna(volatility):
                continue
            
            # SINAL DE COMPRA
            # Breakout banda superior + RSI > 70 + volatilidade adequada
            if (current_price > bb_upper and 
                rsi_val > 70 and 
                volatility > min_volatility):
                buy_signals.append({
                    'x': df.index[i],
                    'y': current_price,
                    'rsi': rsi_val,
                    'reason': f'Breakout Superior: RSI={rsi_val:.1f}'
                })
            
            # Toque banda inferior + RSI < 30 (oversold bounce)
            elif (current_price <= bb_lower * 1.001 and 
                  rsi_val < 30 and 
                  volatility > min_volatility):
                buy_signals.append({
                    'x': df.index[i],
                    'y': current_price,
                    'rsi': rsi_val,
                    'reason': f'Oversold Bounce: RSI={rsi_val:.1f}'
                })
            
            # SINAL DE VENDA (para referência visual)
            elif (current_price < bb_lower and rsi_val < 30):
                sell_signals.append({
                    'x': df.index[i],
                    'y': current_price,
                    'rsi': rsi_val,
                    'reason': f'Oversold: RSI={rsi_val:.1f}'
                })
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def _add_candlestick_plot(self, fig: go.Figure, df: pd.DataFrame, 
                             indicators: Dict, signals: Dict, symbol: str):
        """Adiciona gráfico de candlestick e Bollinger Bands"""
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol,
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['bb_upper'],
                mode='lines',
                name='BB Superior',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['bb_middle'],
                mode='lines',
                name='BB Média (SMA)',
                line=dict(color='blue', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['bb_lower'],
                mode='lines',
                name='BB Inferior',
                line=dict(color='green', width=1, dash='dash'),
                opacity=0.7,
                fill='tonexty',
                fillcolor='rgba(68, 68, 68, 0.1)'
            ),
            row=1, col=1
        )
        
        # Sinais de COMPRA
        if signals['buy_signals']:
            buy_x = [s['x'] for s in signals['buy_signals']]
            buy_y = [s['y'] for s in signals['buy_signals']]
            buy_text = [f"🚀 BUY<br>{s['reason']}" for s in signals['buy_signals']]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode='markers',
                    name='🚀 Sinais de Compra',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='lime',
                        line=dict(color='darkgreen', width=2)
                    ),
                    text=buy_text,
                    hovertemplate='%{text}<br>Preço: $%{y:.6f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Sinais de VENDA (para referência)
        if signals['sell_signals']:
            sell_x = [s['x'] for s in signals['sell_signals']]
            sell_y = [s['y'] for s in signals['sell_signals']]
            sell_text = [f"⚠️ SELL<br>{s['reason']}" for s in signals['sell_signals']]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode='markers',
                    name='⚠️ Sinais de Venda',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(color='darkred', width=2)
                    ),
                    text=sell_text,
                    hovertemplate='%{text}<br>Preço: $%{y:.6f}<extra></extra>'
                ),
                row=1, col=1
            )
    
    def _add_rsi_plot(self, fig: go.Figure, df: pd.DataFrame, 
                     indicators: Dict, signals: Dict, rsi_period: int):
        """Adiciona gráfico do RSI"""
        
        # Linha do RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['rsi'],
                mode='lines',
                name=f'RSI({rsi_period})',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Linhas de referência (70, 50, 30)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
        # Área de sobrecompra/sobrevenda
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, 
                      annotation_text="Sobrecompra", row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, 
                      annotation_text="Sobrevenda", row=2, col=1)
        
        # Marcadores RSI nos sinais
        if signals['buy_signals']:
            buy_rsi_x = [s['x'] for s in signals['buy_signals']]
            buy_rsi_y = [s['rsi'] for s in signals['buy_signals']]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_rsi_x,
                    y=buy_rsi_y,
                    mode='markers',
                    name='RSI nos Sinais BUY',
                    marker=dict(symbol='circle', size=10, color='lime'),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    def _add_volume_plot(self, fig: go.Figure, df: pd.DataFrame, signals: Dict):
        """Adiciona gráfico de volume"""
        
        # Volume colorido baseado na direção do preço
        colors = ['red' if df.iloc[i]['close'] < df.iloc[i]['open'] 
                 else 'green' for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker=dict(color=colors, opacity=0.6),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Destacar volume nos sinais
        if signals['buy_signals']:
            buy_vol_x = [s['x'] for s in signals['buy_signals']]
            buy_vol_y = [df.loc[s['x'], 'volume'] for s in signals['buy_signals'] 
                        if s['x'] in df.index]
            
            if buy_vol_y:  # Se encontrou volumes
                fig.add_trace(
                    go.Bar(
                        x=buy_vol_x,
                        y=buy_vol_y,
                        name='Volume nos Sinais',
                        marker=dict(color='yellow', opacity=0.8),
                        showlegend=False
                    ),
                    row=3, col=1
                )
    
    def _configure_layout(self, fig: go.Figure, symbol: str):
        """Configura layout do gráfico"""
        
        fig.update_layout(
            title=f"📊 {symbol} - Estratégia Bollinger Bands + RSI + Volume",
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_dark"
        )
        
        # Remover range slider do candlestick
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        # Labels dos eixos
        fig.update_yaxes(title_text="Preço ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Volume", row=3, col=1)
        fig.update_xaxes(title_text="Tempo", row=3, col=1)

    def display_strategy_stats(self, symbol: str, period: int = 100):
        """Exibe estatísticas da estratégia"""
        try:
            # CORRIGIDO: sem parâmetro limit
            df = self.market_analyzer.get_market_data(symbol)
            if df is None:
                return
            
            # Limitar aos últimos N candles
            if len(df) > period:
                df = df.tail(period).copy()
            
            # Configurações atuais
            bb_period = st.session_state.get('bb_period', 20)
            bb_std = st.session_state.get('bb_std', 2.0)
            rsi_period = st.session_state.get('rsi_period', 14)
            min_volatility = st.session_state.get('min_volatility', 0.005)
            
            # Calcular indicadores
            indicators = self._calculate_indicators(df, bb_period, bb_std, rsi_period)
            signals = self._identify_signals(df, indicators, min_volatility)
            
            # Valores atuais
            current_price = df.iloc[-1]['close']
            current_rsi = indicators['rsi'].iloc[-1]
            current_bb_upper = indicators['bb_upper'].iloc[-1]
            current_bb_lower = indicators['bb_lower'].iloc[-1]
            current_volatility = indicators['volatility'].iloc[-1]
            
            # Exibir métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Preço Atual", f"${current_price:.6f}")
                st.metric("📊 RSI Atual", f"{current_rsi:.1f}")
            
            with col2:
                st.metric("🔴 BB Superior", f"${current_bb_upper:.6f}")
                st.metric("🔵 BB Média", f"${indicators['bb_middle'].iloc[-1]:.6f}")
            
            with col3:
                st.metric("🟢 BB Inferior", f"${current_bb_lower:.6f}")
                st.metric("📈 Volatilidade", f"{current_volatility:.4f}")
            
            with col4:
                total_buy_signals = len(signals['buy_signals'])
                total_sell_signals = len(signals['sell_signals'])
                st.metric("🚀 Sinais de Compra", total_buy_signals)
                st.metric("⚠️ Sinais de Venda", total_sell_signals)
            
            # Status atual
            if current_rsi > 70 and current_price > current_bb_upper:
                st.success("🚀 **CONDIÇÃO DE COMPRA ATIVA** - Breakout + RSI Alto!")
            elif current_rsi < 30 and current_price <= current_bb_lower * 1.001:
                st.success("🚀 **CONDIÇÃO DE COMPRA ATIVA** - Oversold Bounce!")
            elif current_volatility < min_volatility:
                st.warning("⏳ **AGUARDANDO** - Volatilidade insuficiente")
            else:
                st.info("👀 **MONITORANDO** - Aguardando setup ideal")
                
        except Exception as e:
            st.error(f"Erro ao calcular estatísticas: {e}")
