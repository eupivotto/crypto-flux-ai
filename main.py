"""
AI Trading Bot - Sistema Completo com Monitoramento de Posições
Arquivo principal do Streamlit
"""
import streamlit as st
import time

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
import time

# Configuração da página
st.set_page_config(
    page_title="Crypto Flux AI - Sistema Completo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# No lugar de st.experimental_autorefresh, use:
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

current_time = time.time()
if current_time - st.session_state.last_refresh > 30:  # 30 segundos
    st.session_state.last_refresh = current_time
    st.rerun()


logger = get_logger("MainApp")

# Inicializa keys críticas de sessão (fallback robusto)
def _init_defaults():
    defaults = {
        'selected_symbol': 'BTC/USDT',
        'current_mode': 'Simulado',
        'auto_mode_active': False,
        'auto_sell_enabled': True,
        'scalper_mode': True,
        'today_trades_count': 0,
        'last_trade_time': 0.0,
        'trading_logs': [],
        'last_position_check': 0.0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

# Tenta inicializar via helper, senão aplica defaults
try:
    init_session_state()
except Exception:
    _init_defaults()
else:
    _init_defaults()

def can_execute_trade(position_manager: PositionManager) -> bool:
    """Verifica se pode executar novo trade"""
    trades_count = int(st.session_state.get('today_trades_count', 0))
    last_trade_time = float(st.session_state.get('last_trade_time', 0.0))

    if trades_count >= int(config.MAX_TRADES_PER_DAY):
        return False
    if time.time() - last_trade_time < float(config.MIN_INTERVAL_BETWEEN_TRADES):
        return False
    if position_manager.get_open_positions_count() >= int(config.MAX_OPEN_POSITIONS):
        return False

    return True

def execute_trade_cycle(position_manager: PositionManager, signal_generator: SignalGenerator) -> bool:
    """Executa ciclo completo de trading"""
    if not can_execute_trade(position_manager):
        st.error("❌ Não é possível executar trade no momento")
        return False

    # Símbolo seguro (fallback se AVAILABLE_COINS tiver problema)
    coins = list(getattr(config, 'AVAILABLE_COINS', [])) or ['BTC/USDT']
    selected_symbol = st.session_state.get('selected_symbol', coins[0])
    if selected_symbol not in coins:
        selected_symbol = coins[0]

    signal_generator.scalper_mode = st.session_state.get('scalper_mode', True)
    signal = signal_generator.generate_signal(selected_symbol)

    # Threshold de confiança dinâmico (scalper permite um pouco menor)
    min_conf = 0.7 if not signal_generator.scalper_mode else 0.5

    if not signal or signal.action != 'BUY' or float(getattr(signal, 'confidence', 0.0)) < min_conf:
        st.warning("⏳ Sinal insuficiente para trade")
        return False

    # Obter preço atual
    current_price = position_manager.get_current_price(selected_symbol)
    if not current_price:
        st.error("❌ Não foi possível obter preço atual")
        return False

    # Guardar último preço para a UI sugerir quantidade mínima
    st.session_state['last_price_for_symbol'] = current_price

    # Criar posição usando a quantidade vinda da UI se existir; senão config.TRADE_AMOUNT
    ui_amount = st.session_state.get('ui_trade_amount', None)
    amount_to_use = float(ui_amount) if ui_amount else float(config.TRADE_AMOUNT)

    position = position_manager.create_position(
        selected_symbol,
        amount_to_use,
        current_price,
        st.session_state.get('current_mode', 'Simulado')
    )

    if position:
        st.session_state['today_trades_count'] = int(st.session_state.get('today_trades_count', 0)) + 1
        st.session_state['last_trade_time'] = time.time()
        st.success(f"✅ POSIÇÃO CRIADA: {position.id}")
        st.success(
            f"💰 Entrada: ${position.entry_price:.2f} | "
            f"TP: ${position.take_profit_price:.2f} | SL: ${position.stop_loss_price:.2f}"
        )
        st.success(f"📊 {selected_symbol} | Quantidade: {amount_to_use:.6f}")
        return True
    else:
        st.error("❌ Falha ao criar posição")
        return False

def handle_automation(position_manager: PositionManager, signal_generator: SignalGenerator) -> None:
    """Gerencia o sistema de automação"""
    st.success("🟢 MODO AUTOMÁTICO ATIVO")

    # Monitoramento automático de posições
    if st.session_state.get('auto_sell_enabled', True):
        current_time = time.time()
        if current_time - float(st.session_state.get('last_position_check', 0)) > float(config.POSITION_MONITORING_INTERVAL):
            position_manager.monitor_positions()
            st.session_state.last_position_check = current_time

    # Análise automática para novos trades (com contagem regressiva)
    if can_execute_trade(position_manager):
        placeholder = st.empty()
        for i in range(30, 0, -1):
            placeholder.info(
                f"🤖 Próxima análise automática em {i}s | "
                f"Posições abertas: {position_manager.get_open_positions_count()}"
            )
            time.sleep(1)
        placeholder.empty()

        # Garantir modo scalper atualizado
        signal_generator.scalper_mode = st.session_state.get('scalper_mode', True)

        # Executar ciclo automático
        if execute_trade_cycle(position_manager, signal_generator):
            st.success("🤖 Trade automático executado!")

        # Recarrega para refletir mudanças
        st.rerun()

def main():
    """Função principal da aplicação"""
    # Instâncias dos componentes principais
    position_manager = PositionManager()
    market_analyzer = MarketAnalyzer()
    signal_generator = SignalGenerator()
    dashboard = TradingDashboard()
    displays = PositionDisplays()

    st.title("🚀 AI Trading Bot - Sistema Completo com Monitoramento de Posições")
    st.markdown("Sistema Avançado: Compra → Monitoramento → Venda Automática")
    st.markdown("---")

    # ⚡️ Controle do modo scalper integrado à interface (sidebar)
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
    
    # No main(), após as outras seções do dashboard, adicione:

    # Seção de gráficos (apenas quando estratégia Bollinger+RSI estiver ativa)
    strategy_mode = st.session_state.get('strategy_mode', 'Scalper (Volume/Candle)')
    if "Bollinger" in strategy_mode:
        st.markdown("---")
        dashboard.display_bollinger_charts()


    # Rodapé
    st.markdown("---")
    st.markdown("🚀 AI Trading Bot com Sistema Completo de Posições")
    st.markdown("Funcionalidades: Compra Automática → Monitoramento Contínuo → Venda por TP/SL → Histórico Completo")
    st.caption(
        f"Posições Abertas: {position_manager.get_open_positions_count()} "
        f"| P&L Hoje: ${st.session_state.get('daily_profit_usd', 0.0):.4f} "
        f"| Modo: {st.session_state.get('current_mode', 'Simulado')}"
    )

if __name__ == "__main__":
    main()
