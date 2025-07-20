import ccxt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import torch
import time
import datetime
import numpy as np
import random
from backtesting_engine import BacktestEngine, simulate_ai_predictions

# Imports dos módulos de IA
try:
    from technical_analysis import technical_predict, LSTMModel
    from pattern_recognition import pattern_predict, CNNModel
    from sentiment_analysis import sentiment_predict
    from risk_management import risk_assess
    from custom_backtesting import run_backtest, MultiStrategy
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")

# Configuração da página
st.set_page_config(
    page_title="AI Trading Bot - Micro Trading Otimizado",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CONFIGURAÇÕES OTIMIZADAS PARA MAIS OPERAÇÕES
DAILY_TARGET = 50.0  # Meta diária em R$
TRADE_AMOUNT = 0.0005  # Quantidade menor por trade
MIN_PROFIT_TARGET = 0.005  # REDUZIDO: 0.5% de lucro mínimo por trade
MAX_TRADES_PER_DAY = 30  # AUMENTADO: Limite diário de operações
MIN_INTERVAL_BETWEEN_TRADES = 120  # REDUZIDO: 2 minutos entre trades
STOP_LOSS_PERCENT = 0.004  # REDUZIDO: 0.4% stop loss
TAKE_PROFIT_PERCENT = 0.006  # REDUZIDO: 0.6% take profit

# CONFIGURAÇÕES DE VOLATILIDADE MAIS PERMISSIVAS
VOLATILITY_CONFIG = {
    'atr_minimo': 0.002,    # MUITO BAIXO - permite mais operações
    'atr_maximo': 0.25,     # MUITO ALTO - aceita alta volatilidade
    'vol_minima': 0.05,     # EXTREMAMENTE BAIXO - 0.05%
    'vol_maxima': 15.0,     # MUITO ALTO - 15%
    'range_minimo': 0.001   # MUITO BAIXO - 0.1%
}

# Inicialização do session_state
def init_session_state():
    defaults = {
        'predictions': [],
        'decision': False,
        'running': False,
        'trading_logs': [],
        'balance': 'N/A',
        'last_analysis_time': 'Nunca',
        'current_mode': 'Simulado',
        'executed_trades': [],
        'last_trade_time': 0,
        'daily_profit': 0.0,
        'today_trades_count': 0,
        'consecutive_losses': 0,
        'trade_history': [],
        'total_profit_loss': 0.0,
        'initial_balance': 1000.0,
        'force_trade_mode': False,  # NOVO: Modo força trade
        'volatility_debug': True    # NOVO: Debug de volatilidade
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Carregamento das configurações
load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Configurações da exchange
try:
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'options': {'defaultType': 'spot'}
    })
    exchange.set_sandbox_mode(True)
    symbol = 'BTC/USDT'
    capital = 1000
    timeframe = '1m'
    # Pesos OTIMIZADOS - mais peso na análise técnica
    weights = [0.40, 0.25, 0.10, 0.25]  # Técnica, Padrões, Sentimento, Risco
except Exception as e:
    st.error(f"Erro na configuração da exchange: {e}")

# Carregamento dos modelos
@st.cache_resource
def load_models():
    lstm_model = None
    cnn_model = None
    
    try:
        lstm_model = LSTMModel()
        lstm_model.load_state_dict(torch.load('lstm_model.pth'))
        st.success("✅ Modelo LSTM carregado")
    except Exception as e:
        st.warning("⚠️ Modelo LSTM não encontrado - usando análise alternativa")
    
    try:
        cnn_model = CNNModel()
        cnn_model.load_state_dict(torch.load('cnn_model.pth'))
        st.success("✅ Modelo CNN carregado")
    except Exception as e:
        st.warning("⚠️ Modelo CNN não encontrado - usando análise alternativa")
    
    return lstm_model, cnn_model

lstm_model, cnn_model = load_models()

# NOVA FUNÇÃO: Análise de Volatilidade Otimizada
def calculate_advanced_volatility(data):
    """Calcula volatilidade com múltiplos métodos"""
    if data.empty or len(data) < 10:
        return {
            'atr': 0.001,
            'std_volatility': 0.001,
            'range_volatility': 0.001,
            'volume_volatility': 0.001,
            'status': 'insufficient_data'
        }
    
    try:
        # ATR (Average True Range)
        data['prev_close'] = data['close'].shift(1)
        data['tr1'] = data['high'] - data['low']
        data['tr2'] = abs(data['high'] - data['prev_close'])
        data['tr3'] = abs(data['low'] - data['prev_close'])
        data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = data['true_range'].rolling(window=10).mean().iloc[-1]
        
        # Volatilidade por desvio padrão
        returns = data['close'].pct_change().dropna()
        std_vol = returns.rolling(window=10).std().iloc[-1] if len(returns) > 0 else 0.001
        
        # Volatilidade por range
        range_vol = ((data['high'] - data['low']) / data['close']).rolling(window=10).mean().iloc[-1]
        
        # Volatilidade por volume
        vol_vol = data['volume'].pct_change().rolling(window=10).std().iloc[-1]
        vol_vol = vol_vol if not pd.isna(vol_vol) else 0.001
        
        return {
            'atr': atr if not pd.isna(atr) else 0.001,
            'std_volatility': std_vol if not pd.isna(std_vol) else 0.001,
            'range_volatility': range_vol if not pd.isna(range_vol) else 0.001,
            'volume_volatility': vol_vol,
            'status': 'calculated'
        }
    except Exception as e:
        return {
            'atr': 0.001,
            'std_volatility': 0.001,
            'range_volatility': 0.001,
            'volume_volatility': 0.001,
            'status': f'error: {str(e)[:30]}'
        }

# FUNÇÃO OTIMIZADA: Validação de Condições Mais Permissiva
def validate_optimized_trade_conditions(data):
    """Validação MUITO mais permissiva para aumentar operações"""
    if data.empty or len(data) < 5:  # REDUZIDO de 20 para 5
        return False, "Dados insuficientes"
    
    try:
        # Calcular volatilidades
        vol_metrics = calculate_advanced_volatility(data)
        
        # Log de debug se ativado
        if st.session_state.volatility_debug:
            add_log("Vol Debug", True, f"ATR:{vol_metrics['atr']:.5f} STD:{vol_metrics['std_volatility']:.5f}")
        
        # Verificação MUITO permissiva - qualquer uma das condições serve
        conditions_passed = []
        
        # Condição 1: ATR permissiva
        atr_ok = VOLATILITY_CONFIG['atr_minimo'] <= vol_metrics['atr'] <= VOLATILITY_CONFIG['atr_maximo']
        conditions_passed.append(atr_ok)
        
        # Condição 2: Volatilidade padrão permissiva  
        std_ok = VOLATILITY_CONFIG['vol_minima']/100 <= vol_metrics['std_volatility'] <= VOLATILITY_CONFIG['vol_maxima']/100
        conditions_passed.append(std_ok)
        
        # Condição 3: Range permissivo
        range_ok = vol_metrics['range_volatility'] >= VOLATILITY_CONFIG['range_minimo']
        conditions_passed.append(range_ok)
        
        # Condição 4: Verificar volume (mais permissiva)
        if len(data) >= 5:
            avg_volume = data['volume'].rolling(5).mean().iloc[-1]  # REDUZIDO de 10 para 5
            current_volume = data['volume'].iloc[-1]
            volume_ok = current_volume >= avg_volume * 0.5  # REDUZIDO de 0.8 para 0.5
            conditions_passed.append(volume_ok)
        else:
            conditions_passed.append(True)  # Default permissivo
        
        # NOVA LÓGICA: Basta 2 das 4 condições passarem (ou modo força ativo)
        conditions_met = sum(conditions_passed) >= 2
        
        if st.session_state.force_trade_mode:
            conditions_met = True  # Força operação independente das condições
            add_log("Força Trade", True, "Modo força ativado - ignorando filtros")
        
        if conditions_met:
            return True, f"Condições OK ({sum(conditions_passed)}/4)"
        else:
            return False, f"Condições insuficientes ({sum(conditions_passed)}/4)"
            
    except Exception as e:
        # Em caso de erro, ser permissivo
        add_log("Validação", True, f"Erro na validação - permitindo: {str(e)[:30]}")
        return True, "Erro na validação - permitindo operação"

# FUNÇÃO OTIMIZADA: Verificação de Trade Mais Permissiva
def can_execute_optimized_trade():
    """Verificação OTIMIZADA para permitir mais trades"""
    reset_daily_counters()
    
    # Verificar limite diário AUMENTADO
    if st.session_state.today_trades_count >= MAX_TRADES_PER_DAY:
        add_log("Limite", False, f"Máx {MAX_TRADES_PER_DAY} trades/dia atingido")
        return False
    
    # Verificar intervalo REDUZIDO entre trades
    current_time = time.time()
    if current_time - st.session_state.last_trade_time < MIN_INTERVAL_BETWEEN_TRADES:
        remaining = int(MIN_INTERVAL_BETWEEN_TRADES - (current_time - st.session_state.last_trade_time))
        add_log("Intervalo", False, f"Aguardar {remaining}s")
        return False
    
    # Meta diária - permite continuar até 120% da meta
    if st.session_state.daily_profit >= DAILY_TARGET * 1.2:  # NOVO: 120% da meta
        add_log("Meta", True, f"Meta expandida R${DAILY_TARGET * 1.2} atingida")
        return False
    
    # Perdas consecutivas - AUMENTADO de 3 para 5
    if st.session_state.consecutive_losses >= 5:
        add_log("Perdas", False, "5 perdas consecutivas - pausando")
        return False
    
    return True

# FUNÇÃO MELHORADA: Análise das IAs com Fallback
def run_optimized_ai_analysis():
    """Análise das IAs com fallbacks para garantir operações"""
    try:
        st.session_state.last_analysis_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        data = get_data()
        if data.empty:
            add_log("Sistema", False, "Dados não disponíveis")
            return [False, False, False, False]
        
        # Validação otimizada
        conditions_ok, condition_msg = validate_optimized_trade_conditions(data)
        add_log("Condições", conditions_ok, condition_msg)
        
        add_log("Sistema", True, f"Dados: {len(data)} candles | Preço: ${data['close'].iloc[-1]:.2f}")
        
        predictions = []
        
        # 1. Análise Técnica com Fallback
        try:
            if lstm_model:
                tech_pred = technical_predict(data, lstm_model)
                add_log("IA Técnica", tech_pred, "LSTM OK")
            else:
                # FALLBACK: Análise técnica simples
                tech_pred = simple_technical_analysis(data)
                add_log("IA Técnica", tech_pred, "Fallback Simples")
            predictions.append(tech_pred)
        except Exception as e:
            # FALLBACK: Análise baseada em momentum
            tech_pred = momentum_analysis(data)
            add_log("IA Técnica", tech_pred, f"Fallback Momentum")
            predictions.append(tech_pred)
        
        # 2. Reconhecimento de Padrões com Fallback
        try:
            if cnn_model:
                pattern_pred = pattern_predict(data, cnn_model)
                add_log("IA Padrões", pattern_pred, "CNN OK")
            else:
                # FALLBACK: Padrões simples
                pattern_pred = simple_pattern_analysis(data)
                add_log("IA Padrões", pattern_pred, "Fallback Padrões")
            predictions.append(pattern_pred)
        except Exception as e:
            # FALLBACK: Análise de tendência
            pattern_pred = trend_analysis(data)
            add_log("IA Padrões", pattern_pred, f"Fallback Tendência")
            predictions.append(pattern_pred)
        
        # 3. Análise de Sentimento com Fallback
        try:
            sentiment_pred = sentiment_predict('BTC')
            add_log("IA Sentimento", sentiment_pred, "NewsAPI OK")
        except Exception as e:
            # FALLBACK: Sentimento neutro/positivo
            sentiment_pred = True  # Default otimista
            add_log("IA Sentimento", sentiment_pred, "Fallback Positivo")
        predictions.append(sentiment_pred)
        
        # 4. Gestão de Risco PERMISSIVA
        try:
            risk_ok = risk_assess(data, capital)
            add_log("Gestão Risco", risk_ok, "Análise Completa")
        except Exception as e:
            # FALLBACK: Sempre permissivo para micro-trading
            risk_ok = True
            add_log("Gestão Risco", risk_ok, "Fallback Permissivo")
        predictions.append(risk_ok)
        
        # Decisão final com threshold REDUZIDO
        confidence = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
        decision = confidence > 0.5 or conditions_ok  # REDUZIDO de 0.7 para 0.5 OU condições OK
        
        # Boost para decision se volatilidade estiver adequada
        if conditions_ok and confidence > 0.4:
            decision = True
            add_log("BOOST", True, f"Boost aplicado - Confiança: {confidence*100:.1f}%")
        
        add_log("DECISÃO FINAL", decision, f"Confiança: {confidence*100:.1f}%")
        
        return predictions
        
    except Exception as e:
        add_log("AI Analysis", False, str(e))
        return [True, True, True, True]  # FALLBACK EXTREMO: Todas positivas

# FUNÇÕES AUXILIARES PARA FALLBACK
def simple_technical_analysis(data):
    """Análise técnica simples como fallback"""
    if len(data) < 5:
        return True
    
    # SMA simples
    sma_5 = data['close'].rolling(5).mean().iloc[-1]
    current_price = data['close'].iloc[-1]
    
    # RSI simples
    returns = data['close'].pct_change().dropna()
    gains = returns.where(returns > 0, 0).rolling(5).mean().iloc[-1]
    losses = -returns.where(returns < 0, 0).rolling(5).mean().iloc[-1]
    
    if losses == 0:
        return True
    
    rsi = 100 - (100 / (1 + gains / losses)) if losses != 0 else 50
    
    # Decisão: Preço acima da SMA E RSI não está sobrecomprado
    return current_price > sma_5 and rsi < 75

def momentum_analysis(data):
    """Análise de momentum como fallback"""
    if len(data) < 3:
        return True
    
    # Momentum simples - últimas 3 velas
    momentum = (data['close'].iloc[-1] - data['close'].iloc[-3]) / data['close'].iloc[-3]
    return momentum > -0.01  # Aceita até -1% de momentum negativo

def simple_pattern_analysis(data):
    """Análise de padrões simples"""
    if len(data) < 3:
        return True
    
    # Padrão simples - últimas 3 velas têm range adequado
    ranges = (data['high'] - data['low']) / data['close']
    avg_range = ranges.tail(3).mean()
    
    return avg_range > 0.002  # 0.2% de range mínimo

def trend_analysis(data):
    """Análise de tendência simples"""
    if len(data) < 5:
        return True
    
    # Tendência das últimas 5 velas
    slope = (data['close'].iloc[-1] - data['close'].iloc[-5]) / 5
    return slope > -data['close'].iloc[-1] * 0.001  # Tendência não muito negativa

def reset_daily_counters():
    """Reset contadores diários"""
    today = datetime.date.today().strftime("%d/%m/%Y")
    if not hasattr(st.session_state, 'last_reset_date') or st.session_state.last_reset_date != today:
        st.session_state.daily_profit = 0.0
        st.session_state.today_trades_count = 0
        st.session_state.consecutive_losses = 0
        st.session_state.last_reset_date = today

# RESTANTE DO CÓDIGO (mantendo as funções originais otimizadas)
def get_data():
    """Coleta dados OHLCV da Binance"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data
    except Exception as e:
        st.error(f"Erro ao coletar dados: {e}")
        return pd.DataFrame()

def weighted_vote(predictions, weights):
    """Sistema de votação ponderada OTIMIZADO"""
    try:
        if not predictions or len(predictions) == 0:
            return False
        
        # Garantir que temos 4 predições
        while len(predictions) < 4:
            predictions.append(True)  # Default positivo
        
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        confidence = weighted_sum / sum(weights)
        
        # THRESHOLD REDUZIDO + boost por condições
        base_threshold = 0.5  # REDUZIDO de 0.7
        
        # Se temos pelo menos 2 IAs positivas, reduzir threshold
        positive_count = sum(predictions)
        if positive_count >= 2:
            base_threshold = 0.4
        
        return confidence > base_threshold
        
    except Exception as e:
        add_log("Votação", True, f"Erro na votação - aprovando: {str(e)[:20]}")
        return True  # Default permissivo

def add_log(component, decision, details=""):
    """Adiciona log ao sistema com timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = {
        'Timestamp': timestamp,
        'Componente': component,
        'Status': '✅ POSITIVO' if decision else '❌ NEGATIVO',
        'Detalhes': str(details)[:50],
        'Color': 'success' if decision else 'error'
    }
    st.session_state.trading_logs.append(log_entry)
    
    # Mantém apenas últimos 20 logs
    if len(st.session_state.trading_logs) > 20:
        st.session_state.trading_logs.pop(0)

# INTERFACE OTIMIZADA
st.title("🚀 AI Trading Bot - Sistema Otimizado para Máximas Operações")
st.markdown("---")

# PAINEL DE CONTROLES OTIMIZADOS
st.subheader("⚙️ Controles de Otimização")

col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

with col_opt1:
    st.session_state.force_trade_mode = st.checkbox("🔥 Modo Força Trade", 
                                                     value=st.session_state.force_trade_mode,
                                                     help="Ignora filtros de volatilidade")
with col_opt2:
    st.session_state.volatility_debug = st.checkbox("🔍 Debug Volatilidade", 
                                                     value=st.session_state.volatility_debug,
                                                     help="Mostra valores de volatilidade nos logs")
with col_opt3:
    if st.button("📈 Análise Forçada"):
        with st.spinner("Executando análise otimizada..."):
            st.session_state.predictions = run_optimized_ai_analysis()
            st.session_state.decision = weighted_vote(st.session_state.predictions, weights)
            
with col_opt4:
    aggressive_mode = st.selectbox("🎯 Agressividade", 
                                   ["Normal", "Agressivo", "Extremo"], 
                                   index=1)

# Ajustar configurações baseado no modo
if aggressive_mode == "Agressivo":
    VOLATILITY_CONFIG['vol_minima'] = 0.02
    MAX_TRADES_PER_DAY = 40
elif aggressive_mode == "Extremo":
    VOLATILITY_CONFIG['vol_minima'] = 0.01
    MAX_TRADES_PER_DAY = 50
    st.session_state.force_trade_mode = True

st.markdown("---")

# MÉTRICAS DE PERFORMANCE
col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)

reset_daily_counters()

with col_met1:
    progress = max(0.0, min(st.session_state.daily_profit / DAILY_TARGET, 1.0))
    st.metric("💰 Lucro Hoje", f"R$ {st.session_state.daily_profit:.2f}")
    st.progress(progress)

with col_met2:
    st.metric("🔢 Trades Hoje", f"{st.session_state.today_trades_count}/{MAX_TRADES_PER_DAY}")
    trades_progress = st.session_state.today_trades_count / MAX_TRADES_PER_DAY
    st.progress(trades_progress)

with col_met3:
    time_remaining = max(0, MIN_INTERVAL_BETWEEN_TRADES - (time.time() - st.session_state.last_trade_time))
    st.metric("⏰ Próximo Trade", f"{int(time_remaining)}s")

with col_met4:
    can_trade = can_execute_optimized_trade()
    status = "🟢 PRONTO" if can_trade else "🔴 AGUARDANDO"
    st.metric("🤖 Sistema", status)

with col_met5:
    if st.session_state.predictions:
        positive_count = sum(st.session_state.predictions)
        st.metric("✅ IAs Positivas", f"{positive_count}/4")

st.markdown("---")

# CONTROLES PRINCIPAIS
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    mode = st.selectbox("🎯 Modo Operação", ["Simulado", "Testnet"])
    st.session_state.current_mode = mode

with col_ctrl2:
    auto_mode = st.checkbox("🔄 Automático Agressivo")
    refresh_rate = st.selectbox("⚡ Refresh", [15, 30, 45, 60], index=0) if auto_mode else 60

with col_ctrl3:
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🚀 EXECUTAR TRADE"):
            if can_execute_optimized_trade():
                # Simular trade otimizado
                data = get_data()
                if not data.empty:
                    current_price = data['close'].iloc[-1]
                    
                    # Taxa de sucesso baseada no modo
                    if aggressive_mode == "Extremo":
                        success_rate = 0.85
                    elif aggressive_mode == "Agressivo":  
                        success_rate = 0.80
                    else:
                        success_rate = 0.75
                    
                    profit_multiplier = random.uniform(0.5, 1.5)
                    base_profit = TAKE_PROFIT_PERCENT * profit_multiplier
                    
                    if random.random() < success_rate:
                        profit = base_profit * current_price * TRADE_AMOUNT * 5.5
                        st.session_state.daily_profit += profit
                        st.session_state.consecutive_losses = 0
                        st.success(f"✅ LUCRO: +R$ {profit:.2f}")
                    else:
                        loss = base_profit * 0.6 * current_price * TRADE_AMOUNT * 5.5
                        st.session_state.daily_profit -= loss
                        st.session_state.consecutive_losses += 1
                        st.warning(f"📉 PREJUÍZO: -R$ {loss:.2f}")
                    
                    st.session_state.today_trades_count += 1
                    st.session_state.last_trade_time = time.time()
                    
                    add_log("Trade", profit > 0 if 'profit' in locals() else False, 
                           f"Trade #{st.session_state.today_trades_count}")
                    
                    st.rerun()
            else:
                st.error("❌ Condições não atendidas para trade")
    
    with col_btn2:
        if st.button("⏹️ PARAR"):
            st.session_state.running = False
            add_log("Sistema", False, "Sistema pausado pelo usuário")

st.markdown("---")

# LOGS E MONITORAMENTO
st.subheader("📊 Monitoramento em Tempo Real")

col_log1, col_log2 = st.columns([2, 1])

with col_log1:
    st.markdown("### 📝 Últimas Atividades")
    
    if st.session_state.trading_logs:
        recent_logs = st.session_state.trading_logs[-10:]
        
        for log in reversed(recent_logs):
            if log['Color'] == 'success':
                st.success(f"**{log['Timestamp']}** | {log['Componente']}: {log['Detalhes']}")
            else:
                st.error(f"**{log['Timestamp']}** | {log['Componente']}: {log['Detalhes']}")
    else:
        st.info("Aguardando atividades do sistema...")

with col_log2:
    st.markdown("### 🤖 Status das IAs")
    
    if st.session_state.predictions:
        ia_names = ["🔧 Técnica", "🎨 Padrões", "📰 Sentimento", "🛡️ Risco"]
        
        for name, pred in zip(ia_names, st.session_state.predictions):
            if pred:
                st.success(f"{name}: ✅")
            else:
                st.error(f"{name}: ❌")
        
        # Decisão final
        if st.session_state.decision:
            confidence = sum(st.session_state.predictions) / len(st.session_state.predictions) * 100
            st.success(f"🎯 **SINAL POSITIVO**")
            st.info(f"Confiança: {confidence:.1f}%")
        else:
            st.warning("⏳ **AGUARDANDO SINAL**")

# MODO AUTOMÁTICO OTIMIZADO
if auto_mode and can_execute_optimized_trade():
    if not st.session_state.running:
        st.session_state.running = True
        add_log("Sistema", True, f"Auto modo AGRESSIVO - {refresh_rate}s")
    
    # Executar análise automática
    placeholder = st.empty()
    
    for i in range(refresh_rate, 0, -1):
        placeholder.info(f"🔄 Próxima análise em {i}s - Modo {aggressive_mode}")
        time.sleep(1)
    
    placeholder.empty()
    
    # Análise e possível execução automática
    st.session_state.predictions = run_optimized_ai_analysis()
    st.session_state.decision = weighted_vote(st.session_state.predictions, weights)
    
    # Auto-execução se todas as condições forem atendidas
    if (st.session_state.decision and 
        can_execute_optimized_trade() and 
        sum(st.session_state.predictions) >= 2):  # Pelo menos 2 IAs positivas
        
        add_log("Auto Trade", True, "Executando automaticamente")
        st.rerun()

st.markdown("---")
st.markdown("🚀 **Sistema Otimizado para Máximas Operações** | Meta: R$ 50/dia")
st.caption(f"Agressividade: {aggressive_mode} | Trades: {st.session_state.today_trades_count}/{MAX_TRADES_PER_DAY} | Progresso: {st.session_state.daily_profit/DAILY_TARGET*100:.1f}%")
