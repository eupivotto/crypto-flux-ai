"""
Dashboard principal do AI Trading Bot
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional
from config.settings import config
from utils.logger import get_logger

logger = get_logger(__name__)

class TradingDashboard:
    """Dashboard principal do sistema de trading"""
    
    def __init__(self):
        self.logger = get_logger("TradingDashboard")
    
    def display_main_controls(self):
        """Exibe controles principais do sistema"""
        st.subheader("🎛️ Controles Principais do Sistema")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Seleção de moeda
            current_symbol = st.selectbox(
                "💱 Moeda Ativa", 
                config.AVAILABLE_COINS,
                index=config.AVAILABLE_COINS.index(st.session_state.get('selected_symbol', 'BTC/USDT'))
            )
            if current_symbol != st.session_state.get('selected_symbol'):
                st.session_state.selected_symbol = current_symbol
                self.logger.info(f"Moeda alterada para: {current_symbol}")
        
        with col2:
            # Modo de operação
            trading_mode = st.selectbox(
                "🎯 Modo de Operação", 
                ["Simulado", "Testnet"],
                index=0 if st.session_state.get('current_mode', 'Simulado') == 'Simulado' else 1
            )
            if trading_mode != st.session_state.get('current_mode'):
                st.session_state.current_mode = trading_mode
                self.logger.info(f"Modo alterado para: {trading_mode}")
        
        with col3:
            # Automação
            auto_mode = st.checkbox(
                "🤖 Automação Ativa", 
                value=st.session_state.get('auto_mode_active', False),
                help="Ativa trading automático baseado nas IAs"
            )
            if auto_mode != st.session_state.get('auto_mode_active'):
                st.session_state.auto_mode_active = auto_mode
                status = "ativada" if auto_mode else "desativada"
                self.logger.info(f"Automação {status}")
        
        with col4:
            # Venda automática
            auto_sell = st.checkbox(
                "🔄 Venda Automática", 
                value=st.session_state.get('auto_sell_enabled', True),
                help="Ativa venda automática por TP/SL"
            )
            if auto_sell != st.session_state.get('auto_sell_enabled'):
                st.session_state.auto_sell_enabled = auto_sell
        
        # Status do sistema
        self._display_system_status()
    
    def _display_system_status(self):
        """Exibe status geral do sistema"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "🟢 ATIVO" if st.session_state.get('auto_mode_active', False) else "🔴 PAUSADO"
            st.metric("Status Sistema", status)
        
        with col2:
            trades_hoje = st.session_state.get('today_trades_count', 0)
            st.metric("Trades Hoje", f"{trades_hoje}/{config.MAX_TRADES_PER_DAY}")
        
        with col3:
            last_analysis = st.session_state.get('last_analysis_time', 'Nunca')
            st.metric("Última Análise", last_analysis)
        
        with col4:
            remaining_time = max(0, config.MIN_INTERVAL_BETWEEN_TRADES - 
                               (time.time() - st.session_state.get('last_trade_time', 0)))
            st.metric("Próximo Trade", f"{int(remaining_time)}s")
    
    def display_trading_controls(self, position_manager, signal_generator):
        """Exibe controles de trading"""
        st.subheader("🚀 Controles de Trading")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📈 **ANALISAR MERCADO**", type="primary", use_container_width=True):
                self._execute_market_analysis(signal_generator)
        
        with col2:
            if st.button("🚀 **EXECUTAR TRADE**", type="secondary", use_container_width=True):
                self._execute_manual_trade(position_manager, signal_generator)
        
        with col3:
            if st.button("🔍 **VERIFICAR POSIÇÕES**", use_container_width=True):
                position_manager.monitor_positions()
                st.success("✅ Verificação manual executada")
        
        with col4:
            if st.button("⏹️ **PARAR SISTEMA**", use_container_width=True):
                st.session_state.auto_mode_active = False
                st.session_state.running = False
                st.warning("🛑 Sistema pausado")
        
        # Configurações avançadas
        with st.expander("⚙️ Configurações Avançadas"):
            self._display_advanced_settings()
    
    def _execute_market_analysis(self, signal_generator):
        """Executa análise de mercado"""
        try:
            with st.spinner("Executando análise completa..."):
                symbol = st.session_state.get('selected_symbol', 'BTC/USDT')
                signal = signal_generator.generate_signal(symbol)
                
                if signal:
                    st.session_state.predictions = [
                        signal.technical_score > 0.6,
                        signal.volume_score > 0.6,
                        signal.momentum_score > 0.6,
                        signal.risk_score > 0.6
                    ]
                    st.session_state.decision = signal.action == 'BUY'
                    st.session_state.last_analysis_time = datetime.now().strftime("%H:%M:%S")
                    
                    st.success(f"✅ Análise concluída! Ação: {signal.action} (Confiança: {signal.confidence:.2f})")
                else:
                    st.warning("⚠️ Não foi possível gerar sinal de trading")
                    
        except Exception as e:
            st.error(f"❌ Erro na análise: {e}")
            self.logger.error(f"Erro na análise de mercado: {e}")
    
    def _execute_manual_trade(self, position_manager, signal_generator):
        """Executa trade manual"""
        try:
            symbol = st.session_state.get('selected_symbol', 'BTC/USDT')
            mode = st.session_state.get('current_mode', 'Simulado')
            
            # Verificar se pode executar trade
            trades_count = st.session_state.get('today_trades_count', 0)
            if trades_count >= config.MAX_TRADES_PER_DAY:
                st.error(f"❌ Limite diário de trades atingido ({config.MAX_TRADES_PER_DAY})")
                return
            
            if position_manager.get_open_positions_count() >= config.MAX_OPEN_POSITIONS:
                st.error(f"❌ Máximo de posições abertas atingido ({config.MAX_OPEN_POSITIONS})")
                return
            
            # Gerar sinal
            signal = signal_generator.generate_signal(symbol)
            
            if not signal or signal.action != 'BUY':
                st.error("❌ Sinal de compra não confirmado pelas IAs")
                return
            
            # Obter preço atual
            current_price = position_manager.get_current_price(symbol)
            if not current_price:
                st.error("❌ Não foi possível obter preço atual")
                return
            
            # Criar posição
            position = position_manager.create_position(symbol, config.TRADE_AMOUNT, current_price, mode)
            
            if position:
                st.session_state.today_trades_count += 1
                st.session_state.last_trade_time = time.time()
                
                st.success(f"✅ **TRADE EXECUTADO COM SUCESSO!**")
                st.success(f"📊 Posição: {position.id[-8:]}")
                st.success(f"💰 Entrada: ${position.entry_price:.2f}")
                st.success(f"🎯 Take Profit: ${position.take_profit_price:.2f}")
                st.success(f"🛑 Stop Loss: ${position.stop_loss_price:.2f}")
                
                st.balloons()
            else:
                st.error("❌ Falha na execução do trade")
                
        except Exception as e:
            st.error(f"❌ Erro na execução: {e}")
            self.logger.error(f"Erro na execução manual: {e}")
    
    def _display_advanced_settings(self):
        """Exibe configurações avançadas para scalping"""
        st.markdown("**⚙️ Configurações de Scalping**")
    
        col1, col2 = st.columns(2)
    
        with col1:
            # Quantidade por trade
            trade_amount = st.number_input(
                "💰 Quantidade por Trade (BTC)", 
                min_value=0.0001, 
                max_value=1.0, 
                value=config.TRADE_AMOUNT, 
                step=0.0001,
                format="%.4f", 
                key="trade_amount_setting",
                help="Quantidade de BTC por operação"
            )
        
            # Take Profit para scalping
            take_profit = st.number_input(
                "📈 Take Profit (%)", 
                min_value=0.01,    # Mínimo 0.01% para scalping
                max_value=5.0, 
                value=config.TAKE_PROFIT_PERCENT*100, 
                step=0.01,
                format="%.2f", 
                key="take_profit_setting",
                help="Meta de lucro por operação"
            )
        
        # Intervalo entre trades
            interval = st.number_input(
                "⏱️ Intervalo Entre Trades (s)", 
                min_value=30, 
                max_value=300, 
                value=config.MIN_INTERVAL_BETWEEN_TRADES, 
                step=30,
                key="interval_setting",
                help="Tempo mínimo entre operações"
            )
    
            with col2:
                # Stop Loss para scalping
                stop_loss = st.number_input(
                    "📉 Stop Loss (%)", 
                    min_value=0.01,    # Mínimo 0.01% para scalping
                    max_value=2.0, 
                    value=config.STOP_LOSS_PERCENT*100, 
                    step=0.01,
                    format="%.2f", 
                    key="stop_loss_setting",
                    help="Limite de perda por operação"
                )
        
                # Máximo de posições
                max_positions = st.number_input(
                    "🔄 Máximo de Posições", 
                    min_value=1, 
                    max_value=10, 
                    value=config.MAX_OPEN_POSITIONS, 
                    step=1,
                    key="max_positions_setting",
                    help="Número máximo de posições simultâneas"
                )
        
                # Máximo de trades por dia
                max_trades = st.number_input(
                    "📊 Máximo Trades/Dia", 
                    min_value=10, 
                    max_value=100, 
                    value=config.MAX_TRADES_PER_DAY, 
                    step=5,
                    key="max_trades_setting",
                    help="Limite diário de operações"
                )
    
        # Mostrar relação Risk/Reward
        if take_profit > 0 and stop_loss > 0:
            risk_reward = take_profit / stop_loss
            
            if risk_reward >= 1.5:
                st.success(f"✅ **Relação R/R: 1:{risk_reward:.1f}** - Excelente para scalping")
            elif risk_reward >= 1.2:
                st.warning(f"⚠️ **Relação R/R: 1:{risk_reward:.1f}** - Aceitável")
            else:
                st.error(f"❌ **Relação R/R: 1:{risk_reward:.1f}** - Risco alto")
        
    # Calcular lucro potencial por trade
        if trade_amount > 0 and take_profit > 0:
            # Assumindo BTC a $45,000
            estimated_btc_price = 45000
            profit_per_trade = trade_amount * estimated_btc_price * (take_profit / 100)
            trades_needed = config.DAILY_TARGET_USD / profit_per_trade if profit_per_trade > 0 else 999
            
            st.info(f"💡 **Estimativa:** ${profit_per_trade:.4f} por trade | "
                f"Precisa de ~{trades_needed:.0f} trades para meta diária")


    
    def display_analysis_status(self):
        """Exibe status das análises das IAs"""
        st.subheader("🤖 Status das Inteligências Artificiais")
        
        predictions = st.session_state.get('predictions', [False, False, False, False])
        ia_names = ["📈 Análise Técnica", "📊 Volume", "⚡ Momentum", "🛡️ Gestão de Risco"]
        
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        
        for col, name, prediction in zip(cols, ia_names, predictions):
            with col:
                if prediction:
                    st.success(f"{name}\n✅ POSITIVO")
                else:
                    st.error(f"{name}\n❌ NEGATIVO")
        
        # Decisão final
        decision = st.session_state.get('decision', False)
        positive_count = sum(predictions)
        
        if decision and positive_count >= 3:
            confidence = positive_count / len(predictions) * 100
            st.success(f"🎯 **DECISÃO FINAL: SINAL DE COMPRA** (Confiança: {confidence:.0f}%)")
        elif positive_count >= 2:
            st.warning(f"⚠️ **SINAL FRACO**: {positive_count}/4 IAs positivas")
        else:
            st.error(f"❌ **SINAL NEGATIVO**: {positive_count}/4 IAs positivas")
    
    def display_performance_metrics(self, position_manager):
        """Exibe métricas de performance"""
        st.subheader("📊 Métricas de Performance")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            daily_profit_usd = st.session_state.get('daily_profit_usd', 0.0)
            color = "normal" if daily_profit_usd >= 0 else "inverse"
            st.metric("💰 Lucro Hoje (USD)", f"${daily_profit_usd:.4f}", delta_color=color)
        
        with col2:
            daily_profit_brl = st.session_state.get('daily_profit', 0.0)
            st.metric("💰 Lucro Hoje (BRL)", f"R$ {daily_profit_brl:.2f}")
        
        with col3:
            trades_count = st.session_state.get('today_trades_count', 0)
            progress = trades_count / config.MAX_TRADES_PER_DAY
            st.metric("🔢 Trades Executados", f"{trades_count}/{config.MAX_TRADES_PER_DAY}")
            st.progress(progress)
        
        with col4:
            open_positions = position_manager.get_open_positions_count()
            st.metric("📊 Posições Abertas", f"{open_positions}/{config.MAX_OPEN_POSITIONS}")
        
        with col5:
            unrealized_pnl = position_manager.get_unrealized_pnl()
            st.metric("💎 P&L Não Realizado", f"${unrealized_pnl:.4f}")
        
        # Progress bars para metas
        self._display_progress_bars()
    
    def _display_progress_bars(self):
        """Exibe barras de progresso para metas"""
        col1, col2 = st.columns(2)
        
        with col1:
            daily_profit_usd = st.session_state.get('daily_profit_usd', 0.0)
            progress_usd = max(0.0, min(daily_profit_usd / config.DAILY_TARGET_USD, 1.0))
            st.write("**Meta Diária USD:**")
            st.progress(progress_usd)
            st.caption(f"${daily_profit_usd:.4f} / ${config.DAILY_TARGET_USD:.2f}")
        
        with col2:
            daily_profit_brl = st.session_state.get('daily_profit', 0.0)
            progress_brl = max(0.0, min(daily_profit_brl / config.DAILY_TARGET, 1.0))
            st.write("**Meta Diária BRL:**")
            st.progress(progress_brl)
            st.caption(f"R$ {daily_profit_brl:.2f} / R$ {config.DAILY_TARGET:.2f}")
    
    def display_system_logs(self):
        """Exibe logs do sistema"""
        st.subheader("📝 Logs do Sistema")
        
        # Controles dos logs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_filter = st.selectbox("🔍 Filtrar por:", ["Todos", "Sucessos", "Erros", "Trades"])
        
        with col2:
            log_count = st.selectbox("📊 Mostrar:", [10, 20, 50], index=0)
        
        with col3:
            if st.button("🗑️ Limpar Logs"):
                st.session_state.trading_logs = []
                st.success("Logs limpos!")
        
        # Exibir logs
        logs = st.session_state.get('trading_logs', [])
        
        if logs:
            # Aplicar filtros
            filtered_logs = self._filter_logs(logs, log_filter)
            recent_logs = filtered_logs[-log_count:] if len(filtered_logs) > log_count else filtered_logs
            
            # Mostrar logs em formato de cards
            for log in reversed(recent_logs):
                self._display_log_card(log)
        else:
            st.info("📋 Nenhum log disponível ainda. Execute uma análise ou trade para gerar logs.")
    
    def _filter_logs(self, logs: List[Dict], filter_type: str) -> List[Dict]:
        """Filtra logs por tipo"""
        if filter_type == "Sucessos":
            return [log for log in logs if log.get('decision', False)]
        elif filter_type == "Erros":
            return [log for log in logs if not log.get('decision', True)]
        elif filter_type == "Trades":
            return [log for log in logs if 'Trade' in log.get('component', '')]
        else:
            return logs
    
    def _display_log_card(self, log: Dict):
        """Exibe um log em formato de card"""
        timestamp = log.get('timestamp_str', 'N/A')
        component = log.get('component', 'Sistema')
        details = log.get('details', 'Sem detalhes')
        decision = log.get('decision', False)
        profit_usd = log.get('profit_usd', 0.0)
        auto_mode = log.get('auto_mode', False)
        
        # Ícones e cores
        icon = "🤖" if auto_mode else "👤"
        
        # Informação de lucro
        profit_info = f" | Lucro: ${profit_usd:.4f}" if profit_usd != 0 else ""
        
        # Exibir log
        log_text = f"**{timestamp}** {icon} | {component}: {details}{profit_info}"
        
        if decision:
            st.success(log_text)
        else:
            st.error(log_text)
