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
from core.accounts import get_testnet_balance
from utils.logger import get_logger
from config.exchange_config import exchange

logger = get_logger(__name__)

class TradingDashboard:
    """Dashboard principal do sistema de trading"""

    def __init__(self):
        self.logger = get_logger("TradingDashboard")

    def display_main_controls(self):
        """Exibe controles principais do sistema"""
        st.subheader("🎛️ Controles Principais do Sistema")

        col1, col2, col3, col4 = st.columns(4)

        # Em display_main_controls, adicione:
        with st.sidebar:
            strategy_mode = st.selectbox(
                "🎯 Estratégia Ativa",
                ["Scalper (Volume/Candle)", "Bollinger+RSI+Vol", "Ambas"],
                index=0,
                help="Escolha a estratégia principal do bot"
            )
            st.session_state['strategy_mode'] = strategy_mode
            
            if strategy_mode != "Scalper (Volume/Candle)":
                st.sidebar.write("**⚙️ Configurações Bollinger+RSI**")
                
                bb_period = st.sidebar.slider("Período Bollinger", 10, 30, 20)
                bb_std = st.sidebar.slider("Desvio Padrão", 1.5, 3.0, 2.0, 0.1)
                rsi_period = st.sidebar.slider("Período RSI", 10, 20, 14)
                min_volatility = st.sidebar.slider("Volatilidade Mínima (%)", 0.1, 2.0, 0.5, 0.1) / 100
                
                # Salvar configurações
                st.session_state.update({
                    'bb_period': bb_period,
                    'bb_std': bb_std, 
                    'rsi_period': rsi_period,
                    'min_volatility': min_volatility
                })


        with col1:
            # Seleção de moeda com fallback seguro
            coins = getattr(config, 'AVAILABLE_COINS', ['BTC/USDT'])
            if not coins or not isinstance(coins, list):
                coins = ['BTC/USDT']
            
            selected = st.session_state.get('selected_symbol', coins[0])
            if selected not in coins:
                selected = coins[0]

            current_symbol = st.selectbox(
                "💱 Moeda Ativa",
                coins,
                index=coins.index(selected)
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

        with st.sidebar:
            scalper_mode = st.toggle(
                "🔀 Ativar modo Scalper (micro operações)?",
                value=st.session_state.get('scalper_mode', True),
                help="Se ativo, ignora análise combinada e opera só com sinais de candle/região/volume."
            )
            st.session_state.scalper_mode = scalper_mode
        

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
            remaining_time = max(
                0,
                config.MIN_INTERVAL_BETWEEN_TRADES - (time.time() - st.session_state.get('last_trade_time', 0))
            )
            st.metric("Próximo Trade", f"{int(remaining_time)}s")

    def display_trading_controls(self, position_manager, signal_generator):
        """Exibe controles de trading"""
        st.subheader("🚀 Controles de Trading")

        # Disponibiliza o PM para a UI (para pegar preço atual na seção avançada)
        st.session_state['position_manager'] = position_manager

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

            trades_count = st.session_state.get('today_trades_count', 0)
            if trades_count >= config.MAX_TRADES_PER_DAY:
                st.error(f"❌ Limite diário de trades atingido ({config.MAX_TRADES_PER_DAY})")
                return

            if position_manager.get_open_positions_count() >= config.MAX_OPEN_POSITIONS:
                st.error(f"❌ Máximo de posições abertas atingido ({config.MAX_OPEN_POSITIONS})")
                return

            signal = signal_generator.generate_signal(symbol)
            if not signal or signal.action != 'BUY':
                st.error("❌ Sinal de compra não confirmado pelas IAs")
                return

            current_price = position_manager.get_current_price(symbol)
            if not current_price:
                st.error("❌ Não foi possível obter preço atual")
                return

            # Salvar último preço em sessão (usado na UI para sugestão)
            st.session_state['last_price_for_symbol'] = current_price

            # Usar quantidade escolhida na UI, se disponível
            chosen_amount = st.session_state.get('ui_trade_amount', config.TRADE_AMOUNT)

            # Validação amigável antes do PM (o PM ainda vai checar/ajustar)
            try:
                market = exchange.fetch_market(symbol)
                min_notional = (market.get('limits', {}).get('cost', {}).get('min', 10.0)) or 10.0
            except Exception:
                min_notional = 10.0

            notional = chosen_amount * current_price
            if notional < min_notional:
                st.warning(
                    f"A quantidade atual gera ${notional:.2f}, abaixo do mínimo ${min_notional:.2f}. "
                    f"Ajuste na UI ou ative o autoajuste nas Configurações Avançadas."
                )

            # Criar posição usando a quantidade do UI
            position = position_manager.create_position(symbol, chosen_amount, current_price, mode)

            if position:
                st.session_state.today_trades_count = st.session_state.get('today_trades_count', 0) + 1
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

        # Informações do par e sugestão de quantidade mínima
        symbol = st.session_state.get('selected_symbol', 'BTC/USDT')
        base_asset = symbol.split('/')[0] if '/' in symbol else symbol
        min_notional = 10.0
        current_price = st.session_state.get('last_price_for_symbol', None)

        if current_price is None and 'position_manager' in st.session_state:
            try:
                current_price = st.session_state['position_manager'].get_current_price(symbol)
                if current_price:
                    st.session_state['last_price_for_symbol'] = current_price
            except Exception:
                current_price = None

        try:
            market = exchange.fetch_market(symbol)
            min_notional = (market.get('limits', {}).get('cost', {}).get('min', 10.0)) or 10.0
        except Exception:
            min_notional = 10.0

        min_qty_sugerida = None
        if current_price and current_price > 0:
            min_qty_sugerida = round((min_notional / current_price) * 1.05, 6)  # 5% folga

        info_text = f"Min notional exigido ({symbol}): ${min_notional:.2f}"
        if current_price:
            info_text += f" | Preço atual: ${current_price:.6f}"
        if min_qty_sugerida:
            info_text += f" | Quantidade mínima sugerida: {min_qty_sugerida} {base_asset}"
        st.caption(info_text)

        with col1:
            # Define valor padrão: usa sugestão se a config estiver abaixo
            default_trade_amount = float(config.TRADE_AMOUNT)
            if min_qty_sugerida:
                default_trade_amount = max(default_trade_amount, min_qty_sugerida)

            trade_amount = st.number_input(
                f"💰 Quantidade por Trade ({base_asset})",
                min_value=0.000001,
                max_value=1000.0,
                value=float(default_trade_amount),
                step=0.000001,
                format="%.6f",
                key="trade_amount_setting",
                help=f"Quantidade de {base_asset} por operação"
            )
            st.session_state['ui_trade_amount'] = float(trade_amount)

            take_profit = st.number_input(
                "📈 Take Profit (%)",
                min_value=0.01,
                max_value=5.0,
                value=float(config.TAKE_PROFIT_PERCENT * 100),
                step=0.01,
                format="%.2f",
                key="take_profit_setting",
                help="Meta de lucro por operação"
            )
            interval = st.number_input(
                "⏱️ Intervalo Entre Trades (s)",
                min_value=30,
                max_value=300,
                value=int(max(30, min(300, config.MIN_INTERVAL_BETWEEN_TRADES))),
                step=30,
                key="interval_setting",
                help="Tempo mínimo entre operações"
            )

            auto_adjust = st.toggle(
                "Autoajustar quantidade para cumprir mínimo da corretora",
                value=st.session_state.get('auto_adjust_min_qty', True),
                help="Se ativo, o sistema aumenta automaticamente a quantidade para atingir o notional mínimo do par."
            )
            st.session_state['auto_adjust_min_qty'] = auto_adjust

        with col2:
            stop_loss = st.number_input(
                "📉 Stop Loss (%)",
                min_value=0.01,
                max_value=2.0,
                value=float(config.STOP_LOSS_PERCENT * 100),
                step=0.01,
                format="%.2f",
                key="stop_loss_setting",
                help="Limite de perda por operação"
            )
            max_positions = st.number_input(
                "🔄 Máximo de Posições",
                min_value=1,
                max_value=15,
                value=int(config.MAX_OPEN_POSITIONS),
                step=1,
                key="max_positions_setting",
                help="Número máximo de posições simultâneas"
            )
            max_trades = st.number_input(
                "📊 Máximo Trades/Dia",
                min_value=10,
                max_value=200,
                value=int(config.MAX_TRADES_PER_DAY),
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

        # Estimativa de lucro por trade
        if trade_amount > 0 and take_profit > 0:
            reference_price = current_price if current_price else 45000
            profit_per_trade = float(trade_amount) * float(reference_price) * (float(take_profit) / 100.0)
            trades_needed = config.DAILY_TARGET_USD / profit_per_trade if profit_per_trade > 0 else 999
            st.info(
                f"💡 **Estimativa:** ${profit_per_trade:.4f} por trade | "
                f"Precisa de ~{trades_needed:.0f} trades para meta diária"
            )

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

        col1, col2, col3, col4, col5, col6 = st.columns(6)

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

        with col6:
            saldo = get_testnet_balance()
            if saldo is not None:
                st.metric("💼 Saldo Testnet", f"${saldo:.2f}")
            else:
                st.error("Erro ao obter saldo testnet")

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

        col1, col2, col3 = st.columns(3)

        with col1:
            log_filter = st.selectbox("🔍 Filtrar por:", ["Todos", "Sucessos", "Erros", "Trades"])

        with col2:
            # CORRIGIDO: Lista completa dos logs
            log_count = st.selectbox("📊 Mostrar:", [10, 20, 50], index=0)

        with col3:
            if st.button("🗑️ Limpar Logs"):
                st.session_state.trading_logs = []
                st.success("Logs limpos!")

        logs = st.session_state.get('trading_logs', [])

        if logs:
            filtered_logs = self._filter_logs(logs, log_filter)
            recent_logs = filtered_logs[-log_count:] if len(filtered_logs) > log_count else filtered_logs
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
        icon = "🤖" if auto_mode else "👤"
        profit_info = f" | Lucro: ${profit_usd:.4f}" if profit_usd != 0 else ""
        log_text = f"**{timestamp}** {icon} | {component}: {details}{profit_info}"
        if decision:
            st.success(log_text)
        else:
            st.error(log_text)
            
        # Displays da estrategia de bb
    def display_bollinger_charts(self, symbol: str = None):
        """Exibe gráficos da estratégia Bollinger+RSI"""
        from ui.bollinger_charts import BollingerRSICharts
        
        if symbol is None:
            symbol = st.session_state.get('selected_symbol', 'BTC/USDT')
        
        st.subheader(f"📊 Gráficos de Alerta - Estratégia Bollinger+RSI")
        
        # Controles do gráfico
        col1, col2, col3 = st.columns(3)
        
        with col1:
            period = st.selectbox(
                "📅 Período",
                [50, 100, 200, 500],
                index=1,
                help="Número de candles para análise"
            )
        
        with col2:
            auto_refresh = st.checkbox(
                "🔄 Auto Refresh",
                value=True,
                help="Atualiza gráfico automaticamente"
            )
        
        with col3:
            if st.button("🔄 Atualizar Gráfico", use_container_width=True):
                st.rerun()
        
        # Criar e exibir gráfico
        chart_generator = BollingerRSICharts()
        
        # Estatísticas da estratégia
        with st.expander("📈 Estatísticas da Estratégia (Expandir)", expanded=False):
            chart_generator.display_strategy_stats(symbol, period)
        
        # Gráfico principal
        with st.spinner(f"Carregando gráfico para {symbol}..."):
            fig = chart_generator.create_bollinger_rsi_chart(symbol, period)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Erro ao carregar gráfico. Verifique dados de mercado.")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(30)  # Aguarda 30s
            st.rerun()

