import ccxt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from technical_analysis import technical_predict, LSTMModel
from pattern_recognition import pattern_predict, CNNModel
from sentiment_analysis import sentiment_predict
from risk_management import risk_assess
from custom_backtesting import run_backtest, MultiStrategy
import torch
import time
import datetime

# Inicialização dos session_state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'decision' not in st.session_state:
    st.session_state.decision = False
if 'running' not in st.session_state:
    st.session_state.running = False
if 'trading_logs' not in st.session_state:
    st.session_state.trading_logs = []

load_dotenv()
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Configurações
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

# Modelos
lstm_model = LSTMModel()
try:
    lstm_model.load_state_dict(torch.load('lstm_model.pth'))
except FileNotFoundError:
    st.warning("Modelo LSTM não encontrado. Treine primeiro com test_technical.py.")

cnn_model = CNNModel()
try:
    cnn_model.load_state_dict(torch.load('cnn_model.pth'))
except FileNotFoundError:
    st.warning("Modelo CNN não encontrado. Treine primeiro com test_pattern.py.")

# Pesos para votação
weights = [0.3, 0.25, 0.2, 0.25]  # Tech, Pattern, Sentiment, Risk

def get_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data

def weighted_vote(predictions, weights):
    weighted_sum = sum(p * w for p, w in zip(predictions, weights))
    return weighted_sum / sum(weights) > 0.5

def check_api_permissions():
    try:
        exchange.fetch_balance()
        return None
    except Exception as e:
        return str(e)

def add_log(component, decision, details=""):
    """Adiciona entrada ao log com timestamp"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = {
        'Timestamp': timestamp,
        'Componente': component,
        'Decisão': '🟢 Compra' if decision else '🔴 Aguardar',
        'Detalhes': details
    }
    st.session_state.trading_logs.append(log_entry)
    # Mantém apenas os últimos 50 logs
    if len(st.session_state.trading_logs) > 50:
        st.session_state.trading_logs.pop(0)

def create_market_chart_with_signals():
    """Cria gráfico de candles com sinais das IAs"""
    data = get_data()
    
    fig = go.Figure()
    
    # Gráfico de candles principal
    fig.add_trace(go.Candlestick(
        x=data['timestamp'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='BTC/USDT',
        increasing_line_color='#00FF88',
        decreasing_line_color='#FF6B6B'
    ))
    
    # Adicionar sinais das IAs se disponíveis
    if 'predictions' in st.session_state and st.session_state.predictions:
        current_price = data['close'].iloc[-1]
        current_time = data['timestamp'].iloc[-1]
        
        # Sinal de compra se decisão positiva
        if st.session_state.decision:
            fig.add_trace(go.Scatter(
                x=[current_time],
                y=[current_price],
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='lime'),
                name='Sinal IA: COMPRA'
            ))
    
    fig.update_layout(
        title='📈 Mercado BTC/USDT + Sinais da IA',
        xaxis_title='Tempo',
        yaxis_title='Preço (USDT)',
        height=500,
        template='plotly_dark'
    )
    
    return fig

def main_loop_with_logs(simulated=True, continuous=False):
    """Versão do main_loop que adiciona logs estruturados"""
    perm_check = check_api_permissions()
    if perm_check and "Invalid API-key" in perm_check:
        st.error(f"Erro de permissões detectado: {perm_check}")
        add_log("Sistema", False, "Erro de permissões API")
        return
    
    try:
        balance = exchange.fetch_balance()
        saldo = balance.get('free', {}).get('USDT', 'N/A')
        add_log("Sistema", True, f"Saldo: {saldo} USDT")
    except Exception as e:
        add_log("Sistema", False, f"Erro balance: {str(e)[:50]}")
    
    iteration_count = 0
    while st.session_state.running and continuous and iteration_count < 100:
        try:
            data = get_data()
            current_price = data['close'].iloc[-1]
            add_log("Dados", True, f"Preço atual: ${current_price:.2f}")
        except Exception as e:
            add_log("Dados", False, f"Erro coleta: {str(e)[:30]}")
            time.sleep(10)
            continue
        
        # Análises das IAs
        tech_pred = technical_predict(data, lstm_model)
        add_log("Análise Técnica", tech_pred, "LSTM + Indicadores")
        
        pattern_pred = pattern_predict(data, cnn_model)
        add_log("Padrões Visuais", pattern_pred, "CNN Reconhecimento")
        
        sentiment_pred = sentiment_predict('BTC')
        add_log("Sentimento", sentiment_pred, "NewsAPI + Transformers")
        
        risk_ok = risk_assess(data, capital)
        add_log("Gestão Risco", risk_ok, "VaR + Volatilidade")
        
        predictions = [tech_pred, pattern_pred, sentiment_pred, risk_ok]
        decision = weighted_vote(predictions, weights)
        
        st.session_state.predictions = predictions
        st.session_state.decision = decision
        
        # Log da decisão final
        confidence = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
        add_log("DECISÃO FINAL", decision, f"Confiança: {confidence*100:.1f}%")
        
        # Execução de trades
        if decision and risk_ok:
            if simulated:
                add_log("Trade", True, "SIMULADO - Compra recomendada")
            else:
                try:
                    current_price = data['close'].iloc[-1]
                    amount = 0.001
                    
                    stop_loss = current_price * 0.99
                    take_profit = current_price * 1.02
                    
                    if stop_loss >= current_price:
                        stop_loss = current_price * 0.98
                    if take_profit <= current_price:
                        take_profit = current_price * 1.03
                    
                    params = {'stopLossPrice': stop_loss, 'takeProfitPrice': take_profit}
                    order = exchange.create_market_buy_order(symbol, amount, params)
                    add_log("Trade", True, f"EXECUTADO - ID: {order['id'][:8]}")
                    
                except Exception as e:
                    if "Stop price would trigger immediately" in str(e):
                        try:
                            order = exchange.create_market_buy_order(symbol, amount)
                            add_log("Trade", True, f"EXECUTADO SEM STOP - ID: {order['id'][:8]}")
                        except Exception as e2:
                            add_log("Trade", False, f"Erro: {str(e2)[:30]}")
                    else:
                        add_log("Trade", False, f"Erro: {str(e)[:30]}")
        
        if not continuous:
            break
        time.sleep(60)
        iteration_count += 1

# Dashboard Principal
st.title("🤖 AI Trading Bot Dashboard")

# Controles principais
col_control1, col_control2, col_control3 = st.columns(3)
with col_control1:
    mode = st.selectbox("Modo", ["Simulado", "Testnet"])
with col_control2:
    continuous = st.checkbox("Atualização Contínua")
with col_control3:
    if st.button("🚀 Iniciar Loop"):
        st.session_state.running = True

# Botões de controle
col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    if st.button("⏹️ Parar"):
        st.session_state.running = False
        st.success("Sistema pausado")

with col_btn2:
    if st.button("🗑️ Limpar Logs"):
        st.session_state.trading_logs = []
        st.success("Logs limpos")

with col_btn3:
    if st.button("📊 Análise Única"):
        main_loop_with_logs(simulated=(mode == "Simulado"), continuous=False)

# Layout Principal em Duas Colunas
col1, col2 = st.columns([2, 1])  # Proporção 2:1

# COLUNA ESQUERDA: Gráfico + Métricas
with col1:
    st.subheader("📈 Mercado em Tempo Real")
    
    # Gráfico principal
    chart_fig = create_market_chart_with_signals()
    st.plotly_chart(chart_fig, use_container_width=True)
    
    # Métricas resumidas
    if 'predictions' in st.session_state and st.session_state.predictions:
        st.subheader("🎯 Status das IAs")
        met1, met2, met3, met4 = st.columns(4)
        
        with met1:
            color = "🟢" if st.session_state.predictions[0] else "🔴"
            st.metric("Análise Técnica", f"{color}")
            
        with met2:
            color = "🟢" if st.session_state.predictions[1] else "🔴"
            st.metric("Padrões", f"{color}")
            
        with met3:
            color = "🟢" if st.session_state.predictions[2] else "🔴"
            st.metric("Sentimento", f"{color}")
            
        with met4:
            color = "🟢" if st.session_state.predictions[3] else "🔴"
            st.metric("Risco OK", f"{color}")

# COLUNA DIREITA: Logs em Tabela
with col2:
    st.subheader("📋 Logs do Sistema")
    
    if st.session_state.trading_logs:
        # Criar DataFrame dos logs
        logs_df = pd.DataFrame(st.session_state.trading_logs)
        
        # Exibir tabela com altura fixa e scroll
        st.dataframe(
            logs_df,
            use_container_width=True,
            height=400,  # Altura fixa
            hide_index=True
        )
        
        # Estatísticas rápidas
        st.subheader("📊 Estatísticas")
        total_logs = len(st.session_state.trading_logs)
        compras = sum(1 for log in st.session_state.trading_logs if "🟢" in log['Decisão'])
        
        stat1, stat2 = st.columns(2)
        with stat1:
            st.metric("Total Logs", total_logs)
        with stat2:
            st.metric("Sinais Compra", compras)
    
    else:
        st.info("Nenhum log ainda. Inicie o sistema!")

# Backtesting
with st.expander("🔍 Backtesting Avançado"):
    if st.button("Rodar Backtest"):
        historical_data = get_data()
        results = run_backtest(historical_data, MultiStrategy)
        st.write(results)

# Execução do loop se ativado
if st.session_state.running and continuous:
    main_loop_with_logs(simulated=(mode == "Simulado"), continuous=True)
