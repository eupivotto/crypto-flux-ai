"""
Funções auxiliares para o AI Trading Bot
"""
import streamlit as st
from datetime import datetime, date
from typing import Dict, Any
import time
import random
from utils.logger import get_logger

logger = get_logger(__name__)

def init_session_state():
    """Inicializa todas as variáveis do session_state"""
    defaults = {
        # Estados básicos
        'predictions': [],
        'decision': False,
        'running': False,
        'auto_mode_active': False,
        
        # Dados de trading
        'selected_symbol': 'BTC/USDT',
        'current_mode': 'Simulado',
        'executed_trades': [],
        'trade_history': [],
        
        # Contadores e métricas
        'last_trade_time': 0,
        'today_trades_count': 0,
        'consecutive_losses': 0,
        'consecutive_wins': 0,
        'daily_profit': 0.0,
        'daily_profit_usd': 0.0,
        'total_profit_loss': 0.0,
        
        # Sistema de posições
        'open_positions': [],
        'closed_positions': [],
        'position_counter': 0,
        'last_position_check': 0,
        'auto_sell_enabled': True,
        
        # Configurações
        'force_trade_mode': False,
        'volatility_debug': False,
        'initial_balance': 1000.0,
        'last_reset_date': date.today().strftime("%d/%m/%Y"),
        
        # Análise de mercado
        'last_analysis_time': 'Nunca',
        'coin_analysis_results': {},
        'recommended_coins': [],
        'multi_coin_data': {},
        
        # Dados da exchange
        'balance_data': {},
        'portfolio_value': 0.0,
        'btc_balance': 0.0,
        'usdt_balance': 1000.0,
        'portfolio_history': [],
        
        # Logs
        'trading_logs': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    logger.info("Session state inicializado")

def reset_daily_counters():
    """Reset automático dos contadores diários"""
    today = date.today().strftime("%d/%m/%Y")
    
    if st.session_state.last_reset_date != today:
        st.session_state.daily_profit = 0.0
        st.session_state.daily_profit_usd = 0.0
        st.session_state.today_trades_count = 0
        st.session_state.consecutive_losses = 0
        st.session_state.consecutive_wins = 0
        st.session_state.last_reset_date = today
        
        logger.info(f"Contadores diários resetados para {today}")

def format_currency(value: float, currency: str = "USD") -> str:
    """Formata valores monetários"""
    if currency == "USD":
        return f"${value:.4f}"
    elif currency == "BRL":
        return f"R$ {value:.2f}"
    else:
        return f"{value:.4f}"

def format_percentage(value: float) -> str:
    """Formata percentuais"""
    return f"{value:+.2f}%"

def format_duration(seconds: float) -> str:
    """Formata duração em formato legível"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h{minutes}m"

def calculate_time_remaining(last_time: float, interval: int) -> int:
    """Calcula tempo restante até próxima ação"""
    current_time = time.time()
    elapsed = current_time - last_time
    remaining = max(0, interval - elapsed)
    return int(remaining)

def is_market_hours() -> bool:
    """Verifica se é horário de mercado (crypto 24/7)"""
    return True  # Crypto mercado 24/7

def get_trading_session() -> str:
    """Retorna sessão de trading atual"""
    current_hour = datetime.now().hour
    
    if 6 <= current_hour < 12:
        return "Manhã"
    elif 12 <= current_hour < 18:
        return "Tarde"
    elif 18 <= current_hour < 24:
        return "Noite"
    else:
        return "Madrugada"

def calculate_profit_progress(current_profit: float, target_profit: float) -> float:
    """Calcula progresso em direção à meta"""
    if target_profit <= 0:
        return 0.0
    
    progress = current_profit / target_profit
    return max(0.0, min(1.0, progress))

def generate_position_id() -> str:
    """Gera ID único para posição"""
    timestamp = int(time.time())
    random_suffix = random.randint(1000, 9999)
    return f"POS_{timestamp}_{random_suffix}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisão segura que evita divisão por zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def validate_trade_params(symbol: str, amount: float, price: float) -> Dict[str, Any]:
    """Valida parâmetros de trade"""
    errors = []
    warnings = []
    
    # Validar símbolo
    if not symbol or '/' not in symbol:
        errors.append("Símbolo inválido")
    
    # Validar quantidade
    if amount <= 0:
        errors.append("Quantidade deve ser positiva")
    elif amount < 0.00001:
        warnings.append("Quantidade muito pequena")
    
    # Validar preço
    if price <= 0:
        errors.append("Preço deve ser positivo")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def get_market_status() -> Dict[str, Any]:
    """Retorna status geral do mercado"""
    session = get_trading_session()
    
    # Simular condições de mercado baseadas no horário
    if session == "Manhã":
        volatility = "NORMAL"
        activity = "MODERATE"
    elif session == "Tarde":
        volatility = "HIGH"
        activity = "HIGH"
    elif session == "Noite":
        volatility = "NORMAL"
        activity = "MODERATE"
    else:  # Madrugada
        volatility = "LOW"
        activity = "LOW"
    
    return {
        'session': session,
        'volatility': volatility,
        'activity': activity,
        'timestamp': datetime.now().isoformat()
    }

def cleanup_old_data(max_age_days: int = 30):
    """Limpa dados antigos do session_state"""
    try:
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        # Limpar logs antigos
        if 'trading_logs' in st.session_state:
            original_count = len(st.session_state.trading_logs)
            st.session_state.trading_logs = [
                log for log in st.session_state.trading_logs
                if log.get('timestamp', datetime.now()).timestamp() > cutoff_date
            ]
            cleaned_count = original_count - len(st.session_state.trading_logs)
            if cleaned_count > 0:
                logger.info(f"Removidos {cleaned_count} logs antigos")
        
        # Limpar histórico de trades antigo
        if 'trade_history' in st.session_state:
            original_count = len(st.session_state.trade_history)
            st.session_state.trade_history = [
                trade for trade in st.session_state.trade_history
                if 'saved_at' in trade and 
                datetime.fromisoformat(trade['saved_at'].replace('Z', '+00:00')).timestamp() > cutoff_date
            ]
            cleaned_count = original_count - len(st.session_state.trade_history)
            if cleaned_count > 0:
                logger.info(f"Removidos {cleaned_count} trades antigos")
                
    except Exception as e:
        logger.error(f"Erro na limpeza de dados: {e}")

def get_performance_summary() -> Dict[str, Any]:
    """Retorna resumo de performance"""
    try:
        trades = st.session_state.get('trade_history', [])
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'avg_profit': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        # Calcular métricas
        total_trades = len(trades)
        profitable_trades = len([t for t in trades if t.get('profit_usd', 0) > 0])
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        
        profits = [t.get('profit_usd', 0) for t in trades]
        total_profit = sum(profits)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        best_trade = max(profits) if profits else 0
        worst_trade = min(profits) if profits else 0
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'best_trade': best_trade,
            'worst_trade': worst_trade
        }
        
    except Exception as e:
        logger.error(f"Erro no resumo de performance: {e}")
        return {}
