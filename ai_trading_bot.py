import ccxt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os
import torch
import time
import datetime
import numpy as np
import random
import json
import threading
from typing import Dict, List, Optional, Tuple

# Imports dos módulos de IA
try:
    from technical_analysis import technical_predict, LSTMModel
    from pattern_recognition import pattern_predict, CNNModel
    from sentiment_analysis import sentiment_predict
    from risk_management import risk_assess
    from custom_backtesting import run_backtest, MultiStrategy
    from backtesting_engine import BacktestEngine, simulate_ai_predictions
except ImportError as e:
    st.warning(f"Alguns módulos de IA não foram encontrados: {e}")

# Configuração da página
st.set_page_config(
    page_title="AI Trading Bot - Sistema Completo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CONFIGURAÇÕES GLOBAIS PRINCIPAIS
DAILY_TARGET = 50.0
DAILY_TARGET_USD = 9.09
TRADE_AMOUNT = 0.001
MIN_PROFIT_TARGET = 0.005
MAX_TRADES_PER_DAY = 40
MIN_INTERVAL_BETWEEN_TRADES = 120
STOP_LOSS_PERCENT = 0.0006 
TAKE_PROFIT_PERCENT = 0.001
USD_TO_BRL = 5.5

# CONFIGURAÇÕES DO SISTEMA DE POSIÇÕES
POSITION_MONITORING_INTERVAL = 10  # Verificar posições a cada 15 segundos
ORDER_TIMEOUT = 300
MIN_PROFIT_TO_CLOSE = 0.003
MAX_OPEN_POSITIONS = 3
AUTO_SELL_ENABLED = True

# Lista de moedas disponíveis
AVAILABLE_COINS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
    'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'MATIC/USDT',
    'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'ATOM/USDT', 'FIL/USDT'
]

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
        'selected_symbol': 'BTC/USDT',
        'executed_trades': [],
        'last_trade_time': 0,
        'daily_profit': 0.0,
        'daily_profit_usd': 0.0,
        'today_trades_count': 0,
        'consecutive_losses': 0,
        'trade_history': [],
        'total_profit_loss': 0.0,
        'initial_balance': 1000.0,
        'force_trade_mode': False,
        'volatility_debug': True,
        'portfolio_history': [],
        'balance_data': {},
        'portfolio_value': 0.0,
        'auto_mode_active': False,
        'coin_analysis_results': {},
        'multi_coin_data': {},
        'recommended_coins': [],
        'btc_balance': 0.0,
        'usdt_balance': 1000.0,
        'last_reset_date': datetime.date.today().strftime("%d/%m/%Y"),
        
        # NOVO: Sistema de Posições
        'open_positions': [],
        'closed_positions': [],
        'position_monitoring_active': False,
        'last_position_check': 0,
        'auto_sell_enabled': True,
        'position_counter': 0
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
    timeframe = '1m'
    weights = [0.40, 0.25, 0.10, 0.25]
except Exception as e:
    st.error(f"Erro na configuração da exchange: {e}")

# CLASSES PARA ESTRUTURAS DE DADOS
class Position:
    def __init__(self, symbol, entry_price, amount, mode='Simulado'):
        self.id = f"POS_{int(time.time())}_{random.randint(1000, 9999)}"
        self.symbol = symbol
        self.entry_price = entry_price
        self.amount = amount
        self.mode = mode
        self.entry_time = datetime.datetime.now()
        self.status = 'OPEN'
        self.take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
        self.stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT)
        self.buy_order_id = None
        self.sell_order_id = None
        self.exit_price = None
        self.exit_time = None
        self.pnl_usd = 0.0
        self.pnl_brl = 0.0
        self.exit_reason = None
        
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'amount': self.amount,
            'mode': self.mode,
            'entry_time': self.entry_time.isoformat(),
            'status': self.status,
            'take_profit_price': self.take_profit_price,
            'stop_loss_price': self.stop_loss_price,
            'buy_order_id': self.buy_order_id,
            'sell_order_id': self.sell_order_id,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'pnl_usd': self.pnl_usd,
            'pnl_brl': self.pnl_brl,
            'exit_reason': self.exit_reason
        }

# SISTEMA DE LOGS AVANÇADO
def add_log_advanced(component, decision, details="", profit_usd=0.0, extra_data=None):
    timestamp = datetime.datetime.now()
    
    log_entry = {
        'id': len(st.session_state.trading_logs) + 1,
        'timestamp': timestamp,
        'timestamp_str': timestamp.strftime("%H:%M:%S"),
        'date': timestamp.strftime("%Y-%m-%d"),
        'component': component,
        'decision': decision,
        'status': '✅ POSITIVO' if decision else '❌ NEGATIVO',
        'details': str(details)[:100],
        'color': 'success' if decision else 'error',
        'profit_usd': profit_usd,
        'profit_brl': profit_usd * USD_TO_BRL,
        'extra_data': extra_data or {},
        'auto_mode': st.session_state.auto_mode_active,
        'symbol': st.session_state.selected_symbol
    }
    
    st.session_state.trading_logs.append(log_entry)
    
    if len(st.session_state.trading_logs) > 100:
        st.session_state.trading_logs.pop(0)
    
    save_logs_to_file(log_entry)

def save_logs_to_file(log_entry):
    try:
        log_file = f"trading_logs_{datetime.datetime.now().strftime('%Y-%m-%d')}.json"
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump({k: str(v) for k, v in log_entry.items()}, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"Erro ao salvar log: {e}")

def add_log(component, decision, details=""):
    add_log_advanced(component, decision, details)

# SISTEMA DE MONITORAMENTO DE POSIÇÕES
class PositionManager:
    def __init__(self):
        self.monitoring_active = False
        
    def create_position(self, symbol, amount, entry_price, mode='Simulado'):
        """Cria uma nova posição"""
        try:
            position = Position(symbol, entry_price, amount, mode)
            
            # Executar ordem de compra
            if mode == 'Testnet':
                try:
                    buy_order = exchange.create_market_buy_order(symbol, amount)
                    position.buy_order_id = buy_order['id']
                    position.entry_price = buy_order.get('average', entry_price)
                    
                    add_log_advanced("Buy Order", True, f"Ordem real executada: {buy_order['id'][:8]}", 0, {
                        'order_id': buy_order['id'],
                        'symbol': symbol,
                        'amount': amount,
                        'price': position.entry_price
                    })
                    
                except Exception as e:
                    add_log_advanced("Buy Order", False, f"Erro na ordem real: {str(e)[:50]}")
                    return None
            else:
                # Simular ordem de compra
                add_log_advanced("Buy Order", True, f"Ordem simulada: {symbol} @ ${entry_price:.2f}", 0, {
                    'symbol': symbol,
                    'amount': amount,
                    'price': entry_price,
                    'mode': 'Simulado'
                })
            
            # Adicionar às posições abertas
            st.session_state.open_positions.append(position.to_dict())
            st.session_state.position_counter += 1
            
            add_log_advanced("Position", True, f"Nova posição criada: {position.id}", 0, {
                'position_id': position.id,
                'symbol': symbol,
                'entry_price': position.entry_price,
                'take_profit': position.take_profit_price,
                'stop_loss': position.stop_loss_price
            })
            
            return position
            
        except Exception as e:
            add_log_advanced("Position Error", False, f"Erro ao criar posição: {str(e)}")
            return None
    
    def monitor_positions(self):
        """Monitora todas as posições abertas"""
        if not st.session_state.auto_sell_enabled:
            return
            
        try:
            positions_to_close = []
            
            for pos_dict in st.session_state.open_positions:
                if pos_dict['status'] != 'OPEN':
                    continue
                
                # Obter preço atual
                current_price = self.get_current_price(pos_dict['symbol'])
                if not current_price:
                    continue
                
                position_id = pos_dict['id']
                entry_price = pos_dict['entry_price']
                take_profit = pos_dict['take_profit_price']
                stop_loss = pos_dict['stop_loss_price']
                
                # Verificar take-profit
                if current_price >= take_profit:
                    self.close_position(position_id, current_price, 'TAKE_PROFIT')
                    positions_to_close.append(position_id)
                    continue
                
                # Verificar stop-loss
                if current_price <= stop_loss:
                    self.close_position(position_id, current_price, 'STOP_LOSS')
                    positions_to_close.append(position_id)
                    continue
                
                # Verificar saída por tempo (opcional - 2 horas)
                entry_time = datetime.datetime.fromisoformat(pos_dict['entry_time'])
                if (datetime.datetime.now() - entry_time).seconds > 7200:  # 2 horas
                    pnl_percent = (current_price - entry_price) / entry_price
                    if pnl_percent > 0.002:  # Só sair por tempo se tiver pelo menos 0.2% de lucro
                        self.close_position(position_id, current_price, 'TIME_EXIT')
                        positions_to_close.append(position_id)
            
            # Atualizar timestamp do último check
            st.session_state.last_position_check = time.time()
            
            if positions_to_close:
                add_log_advanced("Monitor", True, f"{len(positions_to_close)} posições fechadas automaticamente")
                
        except Exception as e:
            add_log_advanced("Monitor", False, f"Erro no monitoramento: {str(e)[:50]}")
    
    def close_position(self, position_id, exit_price, exit_reason):
        """Fecha uma posição específica"""
        try:
            # Encontrar a posição
            position_dict = None
            position_index = None
            
            for i, pos in enumerate(st.session_state.open_positions):
                if pos['id'] == position_id:
                    position_dict = pos
                    position_index = i
                    break
            
            if not position_dict:
                add_log_advanced("Close Position", False, f"Posição {position_id} não encontrada")
                return False
            
            # Executar ordem de venda
            if position_dict['mode'] == 'Testnet':
                try:
                    sell_order = exchange.create_market_sell_order(
                        position_dict['symbol'], 
                        position_dict['amount']
                    )
                    exit_price = sell_order.get('average', exit_price)
                    sell_order_id = sell_order['id']
                    
                    add_log_advanced("Sell Order", True, f"Venda real executada: {sell_order_id[:8]}", 0, {
                        'order_id': sell_order_id,
                        'symbol': position_dict['symbol'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason
                    })
                    
                except Exception as e:
                    add_log_advanced("Sell Order", False, f"Erro na venda real: {str(e)[:50]}")
                    return False
            else:
                # Simular ordem de venda
                sell_order_id = f"SIM_{int(time.time())}"
                add_log_advanced("Sell Order", True, f"Venda simulada: {position_dict['symbol']} @ ${exit_price:.2f}")
            
            # Calcular P&L
            entry_price = position_dict['entry_price']
            amount = position_dict['amount']
            pnl_usd = (exit_price - entry_price) * amount
            pnl_brl = pnl_usd * USD_TO_BRL
            
            # Atualizar dados da posição
            position_dict.update({
                'status': 'CLOSED',
                'exit_price': exit_price,
                'exit_time': datetime.datetime.now().isoformat(),
                'exit_reason': exit_reason,
                'sell_order_id': sell_order_id,
                'pnl_usd': pnl_usd,
                'pnl_brl': pnl_brl
            })
            
            # Mover para posições fechadas
            st.session_state.closed_positions.append(position_dict)
            del st.session_state.open_positions[position_index]
            
            # Atualizar lucros globais
            st.session_state.daily_profit_usd += pnl_usd
            st.session_state.daily_profit += pnl_brl
            
            # Registrar no histórico de trades
            trade_record = {
                'id': len(st.session_state.trade_history) + 1,
                'timestamp_str': datetime.datetime.now().strftime("%H:%M:%S"),
                'trade_type': 'COMPLETE_CYCLE',
                'symbol': position_dict['symbol'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'amount': amount,
                'profit_usd': pnl_usd,
                'profit_brl': pnl_brl,
                'status': 'completed',
                'mode': position_dict['mode'],
                'exit_reason': exit_reason,
                'position_id': position_id,
                'duration_minutes': (datetime.datetime.now() - datetime.datetime.fromisoformat(position_dict['entry_time'])).total_seconds() / 60
            }
            
            st.session_state.trade_history.append(trade_record)
            save_trade_to_history(trade_record)
            
            # Log detalhado
            result_type = "LUCRO" if pnl_usd > 0 else "PREJUÍZO"
            add_log_advanced("Position Closed", pnl_usd > 0, 
                           f"{exit_reason}: {result_type} ${pnl_usd:.4f}", pnl_usd, {
                'position_id': position_id,
                'symbol': position_dict['symbol'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'duration_minutes': trade_record['duration_minutes'],
                'exit_reason': exit_reason
            })
            
            # Feedback visual
            if pnl_usd > 0:
                st.success(f"🎯 {exit_reason}: +${pnl_usd:.4f} (+R$ {pnl_brl:.2f}) | {position_dict['symbol']}")
            else:
                st.warning(f"🛑 {exit_reason}: ${pnl_usd:.4f} (R$ {pnl_brl:.2f}) | {position_dict['symbol']}")
            
            return True
            
        except Exception as e:
            add_log_advanced("Close Position", False, f"Erro ao fechar posição: {str(e)}")
            return False
    
    def get_current_price(self, symbol):
        """Obtém preço atual de um símbolo"""
        try:
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            return None
    
    def get_open_positions_count(self):
        """Retorna número de posições abertas"""
        return len([pos for pos in st.session_state.open_positions if pos['status'] == 'OPEN'])
    
    def get_unrealized_pnl(self):
        """Calcula P&L não realizado de todas as posições abertas"""
        total_pnl = 0.0
        
        for pos in st.session_state.open_positions:
            if pos['status'] != 'OPEN':
                continue
                
            current_price = self.get_current_price(pos['symbol'])
            if current_price:
                pnl = (current_price - pos['entry_price']) * pos['amount']
                total_pnl += pnl
        
        return total_pnl

# Instância global do gerenciador
position_manager = PositionManager()

# SISTEMA DE PERSISTÊNCIA
def save_trade_to_history(trade_data):
    trade_data['saved_at'] = datetime.datetime.now().isoformat()
    
    try:
        trades_file = f"trades_database_{datetime.datetime.now().strftime('%Y-%m')}.json"
        with open(trades_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.trade_history, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        add_log_advanced("Database", False, f"Erro ao salvar: {e}")

def save_positions_to_file():
    """Salva posições em arquivo"""
    try:
        positions_file = f"positions_{datetime.datetime.now().strftime('%Y-%m-%d')}.json"
        all_positions = {
            'open_positions': st.session_state.open_positions,
            'closed_positions': st.session_state.closed_positions,
            'last_update': datetime.datetime.now().isoformat()
        }
        with open(positions_file, 'w', encoding='utf-8') as f:
            json.dump(all_positions, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        add_log_advanced("Positions File", False, f"Erro ao salvar posições: {e}")

# FUNÇÕES DE ANÁLISE (simplificadas para focar no sistema de posições)
def get_data():
    try:
        ohlcv = exchange.fetch_ohlcv(st.session_state.selected_symbol, timeframe, limit=100)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data
    except Exception as e:
        st.error(f"Erro ao coletar dados: {e}")
        return pd.DataFrame()

def simple_analysis():
    """Análise simplificada para demonstração"""
    try:
        data = get_data()
        if data.empty:
            return [False, False, False, False]
        
        # Análise técnica simples
        sma_5 = data['close'].rolling(5).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        tech_signal = current_price > sma_5
        
        # Análise de volume
        avg_volume = data['volume'].rolling(10).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_signal = current_volume > avg_volume * 0.8
        
        # Análise de momentum
        momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
        momentum_signal = momentum > -0.01
        
        # Gestão de risco (baseada em posições abertas)
        risk_signal = position_manager.get_open_positions_count() < MAX_OPEN_POSITIONS
        
        predictions = [tech_signal, volume_signal, momentum_signal, risk_signal]
        
        add_log_advanced("Analysis", any(predictions), f"Sinais: {sum(predictions)}/4", 0, {
            'predictions': predictions,
            'current_price': current_price
        })
        
        return predictions
        
    except Exception as e:
        add_log_advanced("Analysis", False, f"Erro na análise: {str(e)}")
        return [True, True, True, True]

def can_execute_trade():
    """Verifica se pode executar novo trade"""
    if st.session_state.today_trades_count >= MAX_TRADES_PER_DAY:
        return False
    
    if time.time() - st.session_state.last_trade_time < MIN_INTERVAL_BETWEEN_TRADES:
        return False
    
    if position_manager.get_open_positions_count() >= MAX_OPEN_POSITIONS:
        return False
    
    return True

def execute_trade_cycle():
    """Executa ciclo completo de trading"""
    if not can_execute_trade():
        st.error("❌ Não é possível executar trade no momento")
        return False
    
    # Executar análise
    predictions = simple_analysis()
    st.session_state.predictions = predictions
    
    # Verificar aprovação
    positive_signals = sum(predictions)
    if positive_signals < 3:
        st.warning(f"⏳ Poucos sinais positivos: {positive_signals}/4")
        return False
    
    # Obter dados atuais
    data = get_data()
    if data.empty:
        st.error("❌ Dados não disponíveis")
        return False
    
    current_price = data['close'].iloc[-1]
    
    # Criar posição
    position = position_manager.create_position(
        st.session_state.selected_symbol,
        TRADE_AMOUNT,
        current_price,
        st.session_state.current_mode
    )
    
    if position:
        st.session_state.today_trades_count += 1
        st.session_state.last_trade_time = time.time()
        
        st.success(f"✅ **POSIÇÃO CRIADA**: {position.id}")
        st.success(f"💰 Entrada: ${position.entry_price:.2f} | TP: ${position.take_profit_price:.2f} | SL: ${position.stop_loss_price:.2f}")
        st.success(f"📊 {st.session_state.selected_symbol} | Quantidade: {TRADE_AMOUNT:.6f}")
        
        return True
    else:
        st.error("❌ Falha ao criar posição")
        return False

# FUNÇÕES DE INTERFACE
def display_position_monitoring():
    """Exibe painel de monitoramento de posições"""
    st.subheader("📌 Sistema de Monitoramento de Posições")
    
    # Controles do sistema
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    
    with col_ctrl1:
        auto_sell = st.checkbox("🔄 Venda Automática", 
                               value=st.session_state.auto_sell_enabled,
                               help="Ativa venda automática por TP/SL")
        st.session_state.auto_sell_enabled = auto_sell
    
    with col_ctrl2:
        if st.button("🔍 Verificar Posições"):
            position_manager.monitor_positions()
            st.success("✅ Verificação manual executada")
    
    with col_ctrl3:
        if st.button("💾 Salvar Posições"):
            save_positions_to_file()
            st.success("✅ Posições salvas em arquivo")
    
    with col_ctrl4:
        st.metric("⏱️ Último Check", f"{int(time.time() - st.session_state.last_position_check)}s atrás")
    
    # Métricas das posições
    col_met1, col_met2, col_met3, col_met4 = st.columns(4)
    
    with col_met1:
        open_count = position_manager.get_open_positions_count()
        st.metric("📊 Posições Abertas", f"{open_count}/{MAX_OPEN_POSITIONS}")
    
    with col_met2:
        closed_count = len(st.session_state.closed_positions)
        st.metric("✅ Posições Fechadas", closed_count)
    
    with col_met3:
        unrealized_pnl = position_manager.get_unrealized_pnl()
        st.metric("💰 P&L Não Realizado", f"${unrealized_pnl:.4f}")
    
    with col_met4:
        total_positions = st.session_state.position_counter
        st.metric("🔢 Total Posições", total_positions)

def display_open_positions():
    """Exibe tabela de posições abertas"""
    st.subheader("📍 Posições Abertas em Tempo Real")
    
    if not st.session_state.open_positions:
        st.info("Nenhuma posição aberta no momento")
        return
    
    positions_data = []
    
    for pos in st.session_state.open_positions:
        if pos['status'] != 'OPEN':
            continue
        
        # Calcular P&L atual
        current_price = position_manager.get_current_price(pos['symbol'])
        if current_price:
            pnl_usd = (current_price - pos['entry_price']) * pos['amount']
            pnl_percent = (current_price - pos['entry_price']) / pos['entry_price'] * 100
            
            # Calcular tempo decorrido
            entry_time = datetime.datetime.fromisoformat(pos['entry_time'])
            duration = (datetime.datetime.now() - entry_time).total_seconds() / 60
            
            positions_data.append({
                'ID': pos['id'][-8:],
                'Símbolo': pos['symbol'],
                'Entrada': f"${pos['entry_price']:.2f}",
                'Atual': f"${current_price:.2f}",
                'Quantidade': f"{pos['amount']:.6f}",
                'P&L USD': f"${pnl_usd:.4f}",
                'P&L %': f"{pnl_percent:+.2f}%",
                'TP': f"${pos['take_profit_price']:.2f}",
                'SL': f"${pos['stop_loss_price']:.2f}",
                'Duração': f"{duration:.0f}m",
                'Modo': pos['mode'],
                'Status': '🟢 LUCRO' if pnl_usd > 0 else '🔴 PREJUÍZO' if pnl_usd < 0 else '⚪ NEUTRO'
            })
    
    if positions_data:
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True, height=300)
        
        # Botão para fechar posição específica
        if st.button("🛑 Fechar Todas as Posições"):
            closed_count = 0
            for pos in st.session_state.open_positions.copy():
                if pos['status'] == 'OPEN':
                    current_price = position_manager.get_current_price(pos['symbol'])
                    if current_price:
                        position_manager.close_position(pos['id'], current_price, 'MANUAL_CLOSE')
                        closed_count += 1
            
            if closed_count > 0:
                st.success(f"✅ {closed_count} posições fechadas manualmente")
                st.rerun()

def display_closed_positions():
    """Exibe tabela de posições fechadas"""
    st.subheader("📋 Histórico de Posições Fechadas")
    
    if not st.session_state.closed_positions:
        st.info("Nenhuma posição fechada ainda")
        return
    
    # Mostrar últimas 20 posições fechadas
    recent_closed = st.session_state.closed_positions[-20:]
    
    closed_data = []
    for pos in recent_closed:
        if pos.get('exit_time'):
            entry_time = datetime.datetime.fromisoformat(pos['entry_time'])
            exit_time = datetime.datetime.fromisoformat(pos['exit_time'])
            duration = (exit_time - entry_time).total_seconds() / 60
            
            closed_data.append({
                'ID': pos['id'][-8:],
                'Símbolo': pos['symbol'],
                'Entrada': f"${pos['entry_price']:.2f}",
                'Saída': f"${pos['exit_price']:.2f}",
                'P&L USD': f"${pos['pnl_usd']:.4f}",
                'P&L BRL': f"R$ {pos['pnl_brl']:.2f}",
                'Duração': f"{duration:.0f}m",
                'Saída': pos['exit_reason'],
                'Modo': pos['mode'],
                'Horário': exit_time.strftime("%H:%M:%S")
            })
    
    if closed_data:
        df_closed = pd.DataFrame(closed_data)
        st.dataframe(df_closed, use_container_width=True, height=400)
        
        # Estatísticas
        total_pnl = sum(pos['pnl_usd'] for pos in recent_closed)
        profitable = len([pos for pos in recent_closed if pos['pnl_usd'] > 0])
        win_rate = (profitable / len(recent_closed) * 100) if recent_closed else 0
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("💰 P&L Total", f"${total_pnl:.4f}")
        with col_stats2:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        with col_stats3:
            st.metric("⚡ Posições", f"{profitable}/{len(recent_closed)}")

# INTERFACE PRINCIPAL
st.title("🚀 AI Trading Bot - Sistema Completo com Monitoramento de Posições")
st.markdown("**Sistema Avançado: Compra → Monitoramento → Venda Automática**")
st.markdown("---")

# Painel de monitoramento
display_position_monitoring()

st.markdown("---")

# CONTROLES PRINCIPAIS
st.subheader("🎛️ Controles de Trading")

col_main1, col_main2, col_main3, col_main4 = st.columns(4)

with col_main1:
    current_symbol = st.selectbox("💱 Moeda", AVAILABLE_COINS, 
                                 index=AVAILABLE_COINS.index(st.session_state.selected_symbol))
    if current_symbol != st.session_state.selected_symbol:
        st.session_state.selected_symbol = current_symbol

with col_main2:
    trading_mode = st.selectbox("🎯 Modo", ["Simulado", "Testnet"])
    st.session_state.current_mode = trading_mode

with col_main3:
    if st.button("🚀 **EXECUTAR TRADE**", type="primary"):
        if execute_trade_cycle():
            st.balloons()
        st.rerun()

with col_main4:
    auto_mode = st.checkbox("🤖 Automação", value=st.session_state.auto_mode_active)
    st.session_state.auto_mode_active = auto_mode

# Status das análises
if st.session_state.predictions:
    st.subheader("🤖 Status das Análises")
    
    col_ia1, col_ia2, col_ia3, col_ia4 = st.columns(4)
    
    ia_names = ["📈 Técnica", "📊 Volume", "⚡ Momentum", "🛡️ Risco"]
    
    for col, name, pred in zip([col_ia1, col_ia2, col_ia3, col_ia4], ia_names, st.session_state.predictions):
        with col:
            if pred:
                st.success(f"{name}\n✅ POSITIVO")
            else:
                st.error(f"{name}\n❌ NEGATIVO")
    
    positive_count = sum(st.session_state.predictions)
    if positive_count >= 3:
        st.success(f"🎯 **SINAL APROVADO**: {positive_count}/4 análises positivas")
    else:
        st.warning(f"⏳ **AGUARDANDO**: {positive_count}/4 análises positivas")

st.markdown("---")

# MÉTRICAS DE PERFORMANCE
col_perf1, col_perf2, col_perf3, col_perf4, col_perf5 = st.columns(5)

with col_perf1:
    st.metric("💰 Lucro Hoje USD", f"${st.session_state.daily_profit_usd:.4f}")

with col_perf2:
    st.metric("💰 Lucro Hoje BRL", f"R$ {st.session_state.daily_profit:.2f}")

with col_perf3:
    st.metric("🔢 Trades Hoje", f"{st.session_state.today_trades_count}/{MAX_TRADES_PER_DAY}")

with col_perf4:
    unrealized = position_manager.get_unrealized_pnl()
    st.metric("📊 P&L Não Realizado", f"${unrealized:.4f}")

with col_perf5:
    total_pnl = st.session_state.daily_profit_usd + unrealized
    st.metric("💎 P&L Total", f"${total_pnl:.4f}")

st.markdown("---")

# TABELAS DE POSIÇÕES
col_pos1, col_pos2 = st.columns(2)

with col_pos1:
    display_open_positions()

with col_pos2:
    display_closed_positions()

st.markdown("---")

# LOGS DO SISTEMA
st.subheader("📝 Logs do Sistema")

if st.session_state.trading_logs:
    recent_logs = st.session_state.trading_logs[-10:]
    
    for log in reversed(recent_logs):
        profit_info = f" (${log.get('profit_usd', 0):.4f})" if log.get('profit_usd', 0) != 0 else ""
        auto_indicator = " 🤖" if log.get('auto_mode', False) else ""
        
        if log['color'] == 'success':
            st.success(f"**{log['timestamp_str']}**{auto_indicator} | {log['component']}: {log['details']}{profit_info}")
        else:
            st.error(f"**{log['timestamp_str']}**{auto_indicator} | {log['component']}: {log['details']}{profit_info}")
else:
    st.info("Execute um trade para ver os logs")

# AUTOMAÇÃO
if st.session_state.auto_mode_active:
    st.success("🟢 **MODO AUTOMÁTICO ATIVO**")
    
    # Monitoramento automático de posições
    if st.session_state.auto_sell_enabled:
        current_time = time.time()
        if current_time - st.session_state.last_position_check > POSITION_MONITORING_INTERVAL:
            position_manager.monitor_positions()
    
    # Análise automática para novos trades
    if can_execute_trade():
        placeholder = st.empty()
        
        for i in range(30, 0, -1):
            placeholder.info(f"🤖 Próxima análise automática em {i}s | Posições: {position_manager.get_open_positions_count()}")
            time.sleep(1)
        
        placeholder.empty()
        
        # Executar ciclo automático
        if execute_trade_cycle():
            st.success("🤖 Trade automático executado!")
        
        st.rerun()

st.markdown("---")
st.markdown("🚀 **AI Trading Bot com Sistema Completo de Posições**")
st.markdown("**Funcionalidades:** Compra Automática → Monitoramento Contínuo → Venda por TP/SL → Histórico Completo")
st.caption(f"Posições Abertas: {position_manager.get_open_positions_count()} | P&L Hoje: ${st.session_state.daily_profit_usd:.4f} | Modo: {st.session_state.current_mode}")
