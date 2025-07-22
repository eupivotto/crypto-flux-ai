"""
Displays específicos para o AI Trading Bot
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class PositionDisplays:
    """Classe para exibir informações de posições"""
    
    def __init__(self):
        self.logger = get_logger("PositionDisplays")
    
    def display_positions_overview(self, position_manager):
        """Exibe visão geral das posições"""
        st.subheader("📊 Visão Geral das Posições")
        
        # Métricas gerais
        self._display_position_metrics(position_manager)
        
        st.markdown("---")
        
        # Layout em duas colunas
        col1, col2 = st.columns(2)
        
        with col1:
            self.display_open_positions(position_manager)
        
        with col2:
            self.display_closed_positions()
    
    def _display_position_metrics(self, position_manager):
        """Exibe métricas das posições"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            open_count = position_manager.get_open_positions_count()
            st.metric("📍 Posições Abertas", open_count)
        
        with col2:
            closed_count = len(st.session_state.get('closed_positions', []))
            st.metric("✅ Posições Fechadas", closed_count)
        
        with col3:
            unrealized_pnl = position_manager.get_unrealized_pnl()
            color = "normal" if unrealized_pnl >= 0 else "inverse"
            st.metric("💰 P&L Não Realizado", f"${unrealized_pnl:.4f}", delta_color=color)
        
        with col4:
            total_positions = st.session_state.get('position_counter', 0)
            st.metric("📈 Total Posições", total_positions)
    
    def display_open_positions(self, position_manager):
        """Exibe posições abertas"""
        st.subheader("📍 Posições Abertas")
        
        open_positions = st.session_state.get('open_positions', [])
        
        if not open_positions:
            st.info("Nenhuma posição aberta no momento")
            return
        
        # Filtrar apenas posições realmente abertas
        active_positions = [pos for pos in open_positions if pos.get('status') == 'OPEN']
        
        if not active_positions:
            st.info("Nenhuma posição ativa no momento")
            return
        
        # Preparar dados para tabela
        positions_data = []
        
        for pos in active_positions:
            # Obter preço atual
            current_price = position_manager.get_current_price(pos['symbol'])
            
            if current_price:
                # Calcular P&L atual
                entry_price = pos['entry_price']
                amount = pos['amount']
                pnl_usd = (current_price - entry_price) * amount
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                
                # Calcular tempo decorrido
                entry_time = datetime.fromisoformat(pos['entry_time'])
                duration = (datetime.now() - entry_time).total_seconds() / 60
                
                # Status baseado no P&L
                if pnl_usd > 0:
                    status = "🟢 LUCRO"
                elif pnl_usd < 0:
                    status = "🔴 PREJUÍZO"
                else:
                    status = "⚪ NEUTRO"
                
                positions_data.append({
                    'ID': pos['id'][-8:],
                    'Símbolo': pos['symbol'],
                    'Entrada': f"${entry_price:.2f}",
                    'Atual': f"${current_price:.2f}",
                    'Quantidade': f"{amount:.6f}",
                    'P&L USD': f"${pnl_usd:.4f}",
                    'P&L %': f"{pnl_percent:+.2f}%",
                    'TP': f"${pos['take_profit_price']:.2f}",
                    'SL': f"${pos['stop_loss_price']:.2f}",
                    'Duração': f"{duration:.0f}m",
                    'Modo': pos['mode'],
                    'Status': status
                })
        
        if positions_data:
            # Exibir tabela
            df_positions = pd.DataFrame(positions_data)
            st.dataframe(df_positions, use_container_width=True, height=300)
            
            # Controles de posições
            self._display_position_controls(position_manager, active_positions)
        else:
            st.warning("Erro ao calcular dados das posições")
    
    def _display_position_controls(self, position_manager, active_positions):
        """Exibe controles para posições abertas"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Verificar Posições", use_container_width=True):
                position_manager.monitor_positions()
                st.success("✅ Verificação executada")
        
        with col2:
            if st.button("🛑 Fechar Todas", use_container_width=True):
                closed_count = 0
                for pos in active_positions:
                    current_price = position_manager.get_current_price(pos['symbol'])
                    if current_price:
                        success = position_manager.close_position(pos['id'], current_price, 'MANUAL_CLOSE')
                        if success:
                            closed_count += 1
                
                if closed_count > 0:
                    st.success(f"✅ {closed_count} posições fechadas")
                    st.rerun()
        
        with col3:
            if st.button("💾 Salvar Estado", use_container_width=True):
                # Implementar salvamento de estado
                st.info("Estado das posições salvo")
    
    def display_closed_positions(self):
        """Exibe posições fechadas"""
        st.subheader("📋 Posições Fechadas Recentes")
        
        closed_positions = st.session_state.get('closed_positions', [])
        
        if not closed_positions:
            st.info("Nenhuma posição fechada ainda")
            return
        
        # Mostrar últimas 10 posições fechadas
        recent_closed = closed_positions[-10:] if len(closed_positions) > 10 else closed_positions
        
        closed_data = []
        for pos in recent_closed:
            if pos.get('exit_time'):
                entry_time = datetime.fromisoformat(pos['entry_time'])
                exit_time = datetime.fromisoformat(pos['exit_time'])
                duration = (exit_time - entry_time).total_seconds() / 60
                
                # Status da saída
                exit_reason = pos.get('exit_reason', 'UNKNOWN')
                if exit_reason == 'TAKE_PROFIT':
                    exit_icon = "🎯"
                elif exit_reason == 'STOP_LOSS':
                    exit_icon = "🛑"
                elif exit_reason == 'MANUAL_CLOSE':
                    exit_icon = "👤"
                else:
                    exit_icon = "⏰"
                
                closed_data.append({
                    'ID': pos['id'][-8:],
                    'Símbolo': pos['symbol'],
                    'Entrada': f"${pos['entry_price']:.2f}",
                    'Saída': f"${pos['exit_price']:.2f}",
                    'P&L USD': f"${pos['pnl_usd']:.4f}",
                    'P&L BRL': f"R$ {pos['pnl_brl']:.2f}",
                    'Duração': f"{duration:.0f}m",
                    'Saída': f"{exit_icon} {exit_reason}",
                    'Modo': pos['mode'],
                    'Horário': exit_time.strftime("%H:%M:%S")
                })
        
        if closed_data:
            # Exibir tabela
            df_closed = pd.DataFrame(closed_data)
            st.dataframe(df_closed, use_container_width=True, height=300)
            
            # Estatísticas rápidas
            self._display_closed_statistics(recent_closed)
        else:
            st.warning("Erro ao processar posições fechadas")
    
    def _display_closed_statistics(self, closed_positions):
        """Exibe estatísticas das posições fechadas"""
        if not closed_positions:
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_pnl = sum(pos.get('pnl_usd', 0) for pos in closed_positions)
            color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("💰 P&L Total", f"${total_pnl:.4f}", delta_color=color)
        
        with col2:
            profitable = len([pos for pos in closed_positions if pos.get('pnl_usd', 0) > 0])
            win_rate = (profitable / len(closed_positions) * 100) if closed_positions else 0
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            avg_duration = sum(
                (datetime.fromisoformat(pos['exit_time']) - datetime.fromisoformat(pos['entry_time'])).total_seconds() / 60
                for pos in closed_positions if pos.get('exit_time')
            ) / len(closed_positions) if closed_positions else 0
            st.metric("⏱️ Duração Média", f"{avg_duration:.0f}m")
    
    def display_performance_chart(self):
        """Exibe gráfico de performance das posições"""
        st.subheader("📈 Gráfico de Performance")
        
        closed_positions = st.session_state.get('closed_positions', [])
        
        if len(closed_positions) < 2:
            st.info("Precisa de pelo menos 2 posições fechadas para gerar gráfico")
            return
        
        try:
            # Preparar dados para gráfico
            df = pd.DataFrame(closed_positions)
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['cumulative_pnl'] = df['pnl_usd'].cumsum()
            
            # Criar gráfico
            fig = go.Figure()
            
            # Linha de P&L cumulativo
            fig.add_trace(go.Scatter(
                x=df['exit_time'],
                y=df['cumulative_pnl'],
                mode='lines+markers',
                name='P&L Cumulativo',
                line=dict(color='green' if df['cumulative_pnl'].iloc[-1] > 0 else 'red', width=2),
                marker=dict(size=6)
            ))
            
            # Linha zero
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Breakeven")
            
            # Configurar layout
            fig.update_layout(
                title="Evolução do P&L Cumulativo",
                xaxis_title="Tempo",
                yaxis_title="P&L Cumulativo (USD)",
                height=400,
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")
            self.logger.error(f"Erro no gráfico de performance: {e}")
    
    def display_position_heatmap(self):
        """Exibe heatmap de performance por horário"""
        st.subheader("🔥 Heatmap de Performance por Horário")
        
        closed_positions = st.session_state.get('closed_positions', [])
        
        if len(closed_positions) < 5:
            st.info("Precisa de pelo menos 5 posições para gerar heatmap")
            return
        
        try:
            # Processar dados por horário
            hourly_performance = {}
            
            for pos in closed_positions:
                if pos.get('entry_time'):
                    entry_time = datetime.fromisoformat(pos['entry_time'])
                    hour = entry_time.hour
                    pnl = pos.get('pnl_usd', 0)
                    
                    if hour not in hourly_performance:
                        hourly_performance[hour] = []
                    hourly_performance[hour].append(pnl)
            
            # Calcular médias por horário
            hourly_avg = {hour: sum(pnls)/len(pnls) for hour, pnls in hourly_performance.items()}
            
            if hourly_avg:
                # Criar DataFrame para heatmap
                hours = list(range(24))
                values = [hourly_avg.get(hour, 0) for hour in hours]
                
                # Exibir métricas por horário
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_hour = max(hourly_avg, key=hourly_avg.get)
                    st.metric("🕐 Melhor Horário", f"{best_hour:02d}:00")
                
                with col2:
                    worst_hour = min(hourly_avg, key=hourly_avg.get)
                    st.metric("🕐 Pior Horário", f"{worst_hour:02d}:00")
                
                with col3:
                    avg_pnl = sum(hourly_avg.values()) / len(hourly_avg)
                    st.metric("💰 P&L Médio/Hora", f"${avg_pnl:.4f}")
                
                # Gráfico de barras por horário
                fig = go.Figure(data=[
                    go.Bar(x=hours, y=values, 
                          marker_color=['green' if v > 0 else 'red' for v in values])
                ])
                
                fig.update_layout(
                    title="Performance Média por Horário do Dia",
                    xaxis_title="Hora do Dia",
                    yaxis_title="P&L Médio (USD)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao gerar heatmap: {e}")
            self.logger.error(f"Erro no heatmap: {e}")
