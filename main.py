"""
AI Trading Bot - Sistema Completo com Monitoramento de Posições
Arquivo principal do Streamlit
"""
import streamlit as st
import time
from datetime import datetime

# Importações dos módulos do projeto
from config.settings import config
from config.exchange_config import exchange
from core.position_manager import PositionManager
from analysis.market_analyzer import MarketAnalyzer
from analysis.signals import SignalGenerator
from ui.dashboard import TradingDashboard
from ui.displays import PositionDisplays
from utils.logger import get_logger
from utils.helpers import init_session_state

# Configuração da página
st.set_page_config(
    page_title="Crypto Flux AI - Sistema Completo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.experimental_rerun_interval = 30 

# Inicialização
logger = get_logger("MainApp")
init_session_state()

# Instâncias dos componentes principais
position_manager = PositionManager()
market_analyzer = MarketAnalyzer()
signal_generator = SignalGenerator()
dashboard = TradingDashboard()
displays = PositionDisplays()

def main():
    """Função principal da aplicação"""

    st.title("🚀 AI Trading Bot - Sistema Completo com Monitoramento de Posições")
    st.markdown("**Sistema Avançado: Compra → Monitoramento → Venda Automática**")
    st.markdown("---")

    # ⚡️ Controle do modo scalper integrado à interface
    # (pode estar no sidebar, nas configs avançadas do dashboard ou aqui)
    st.sidebar.divider()
    scalper_mode = st.sidebar.toggle(
        "⚡️ Ativar modo Scalper (micro-operações)?",
        value=st.session_state.get('scalper_mode', True),
        help="Scalper: opera sinais rápidos baseados em volume, região e candle."
    )
    st.session_state['scalper_mode'] = scalper_mode
    signal_generator.scalper_mode = scalper_mode

    st.sidebar.info(
        f"Modo operacional: {'Scalper (micro operações rápidas)' if scalper_mode else 'Convencional (sinais analisados)'}"
    )
    st.sidebar.divider()

    # Dashboard principal
    dashboard.display_main_controls()
    st.markdown("---")
    dashboard.display_trading_controls(position_manager, signal_generator)
    dashboard.display_analysis_status()
    st.markdown("---")
    dashboard.display_performance_metrics(position_manager)
    st.markdown("---")
    displays.display_positions_overview(position_manager)
    st.markdown("---")
    dashboard.display_system_logs()

    # Sistema de automação
    if st.session_state.get('auto_mode_active', False):
        handle_automation(position_manager, signal_generator)

    # Rodapé
    st.markdown("---")
    st.markdown("🚀 **AI Trading Bot com Sistema Completo de Posições**")
    st.markdown("**Funcionalidades:** Compra Automática → Monitoramento Contínuo → Venda por TP/SL → Histórico Completo")
    st.caption(f"Posições Abertas: {position_manager.get_open_positions_count()} | P&L Hoje: ${st.session_state.get('daily_profit_usd', 0.0):.4f} | Modo: {st.session_state.get('current_mode', 'Simulado')}")

def handle_automation(position_manager: PositionManager, signal_generator: SignalGenerator):
    """Gerencia o sistema de automação"""
    st.success("🟢 **MODO AUTOMÁTICO ATIVO**")

    # Monitoramento automático de posições
    if st.session_state.get('auto_sell_enabled', True):
        current_time = time.time()
        if current_time - getattr(st.session_state, 'last_position_check', 0) > config.POSITION_MONITORING_INTERVAL:
            position_manager.monitor_positions()
            st.session_state.last_position_check = current_time

    # Análise automática para novos trades
    if can_execute_trade(position_manager):
        placeholder = st.empty()
        for i in range(30, 0, -1):
            placeholder.info(f"🤖 Próxima análise automática em {i}s | Posições: {position_manager.get_open_positions_count()}")
            time.sleep(1)
        placeholder.empty()

        # Atualiza modo scalper antes do ciclo!
        signal_generator.scalper_mode = st.session_state.get('scalper_mode', True)

        # Executar ciclo automático
        if execute_trade_cycle(position_manager, signal_generator):
            st.success("🤖 Trade automático executado!")

        st.rerun()

def can_execute_trade(position_manager: PositionManager) -> bool:
    """Verifica se pode executar novo trade"""
    trades_count = getattr(st.session_state, 'today_trades_count', 0)
    last_trade_time = getattr(st.session_state, 'last_trade_time', 0)

    if trades_count >= config.MAX_TRADES_PER_DAY:
        return False
    if time.time() - last_trade_time < config.MIN_INTERVAL_BETWEEN_TRADES:
        return False
    if position_manager.get_open_positions_count() >= config.MAX_OPEN_POSITIONS:
        return False

    return True

def execute_trade_cycle(position_manager: PositionManager, signal_generator: SignalGenerator) -> bool:
    """Executa ciclo completo de trading"""
    if not can_execute_trade(position_manager):
        st.error("❌ Não é possível executar trade no momento")
        return False

    signal_generator.scalper_mode = st.session_state.get('scalper_mode', True)
    signal = signal_generator.generate_signal(st.session_state.selected_symbol)

    # Scalper pode operar com thresholds mais baixos!
    # Ajusta a confiança mínima conforme o modo, se desejar.
    min_conf = 0.7 if not signal_generator.scalper_mode else 0.5

    if not signal or signal.action != 'BUY' or signal.confidence < min_conf:
        st.warning(f"⏳ Sinal insuficiente para trade")
        return False

    # Obter preço atual
    current_price = position_manager.get_current_price(st.session_state.selected_symbol)
    if not current_price:
        st.error("❌ Não foi possível obter preço atual")
        return False

    # Criar posição
    position = position_manager.create_position(
        st.session_state.selected_symbol,
        config.TRADE_AMOUNT,
        current_price,
        st.session_state.current_mode
    )

    if position:
        if 'today_trades_count' not in st.session_state:
            st.session_state.today_trades_count = 0
        st.session_state.today_trades_count += 1
        st.session_state.last_trade_time = time.time()
        st.success(f"✅ **POSIÇÃO CRIADA**: {position.id}")
        st.success(f"💰 Entrada: ${position.entry_price:.2f} | TP: ${position.take_profit_price:.2f} | SL: ${position.stop_loss_price:.2f}")
        st.success(f"📊 {st.session_state.selected_symbol} | Quantidade: {config.TRADE_AMOUNT:.6f}")
        return True
    else:
        st.error("❌ Falha ao criar posição")
        return False

if __name__ == "__main__":
    main()
