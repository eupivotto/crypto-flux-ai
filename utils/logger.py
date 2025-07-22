"""
Sistema de logs avançado do AI Trading Bot
"""
import logging
import datetime
import json
import streamlit as st
from typing import Dict, Any, Optional

class TradingLogger:
    """Logger personalizado para trading"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """Configura o logger"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, extra_data: Optional[Dict] = None):
        self.logger.info(message)
        self._add_to_streamlit_logs("INFO", True, message, extra_data)
    
    def error(self, message: str, extra_data: Optional[Dict] = None):
        self.logger.error(message)
        self._add_to_streamlit_logs("ERROR", False, message, extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict] = None):
        self.logger.warning(message)
        self._add_to_streamlit_logs("WARNING", False, message, extra_data)
    
    def _add_to_streamlit_logs(self, level: str, success: bool, message: str, extra_data: Optional[Dict]):
        """Adiciona log ao sistema do Streamlit"""
        if 'trading_logs' not in st.session_state:
            st.session_state.trading_logs = []
        
        timestamp = datetime.datetime.now()
        
        log_entry = {
            'id': len(st.session_state.trading_logs) + 1,
            'timestamp': timestamp,
            'timestamp_str': timestamp.strftime("%H:%M:%S"),
            'date': timestamp.strftime("%Y-%m-%d"),
            'component': self.name,
            'level': level,
            'decision': success,
            'status': '✅ SUCCESS' if success else '❌ ERROR',
            'details': str(message)[:100],
            'color': 'success' if success else 'error',
            'extra_data': extra_data or {},
            'auto_mode': getattr(st.session_state, 'auto_mode_active', False),
            'symbol': getattr(st.session_state, 'selected_symbol', 'N/A')
        }
        
        st.session_state.trading_logs.append(log_entry)
        
        # Manter apenas últimos 100 logs
        if len(st.session_state.trading_logs) > 100:
            st.session_state.trading_logs.pop(0)
        
        # Salvar em arquivo
        self._save_to_file(log_entry)
    
    def _save_to_file(self, log_entry: Dict):
        """Salva log em arquivo"""
        try:
            log_file = f"data/trading_logs_{datetime.datetime.now().strftime('%Y-%m-%d')}.json"
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump({k: str(v) for k, v in log_entry.items()}, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"Erro ao salvar log: {e}")

def get_logger(name: str) -> TradingLogger:
    """Retorna instância do logger"""
    return TradingLogger(name)

# Função de compatibilidade
def add_log_advanced(component: str, decision: bool, details: str = "", profit_usd: float = 0.0, extra_data: Dict = None):
    """Função de compatibilidade com o código existente"""
    logger = get_logger(component)
    
    if decision:
        logger.info(details, extra_data)
    else:
        logger.error(details, extra_data)
    
    # Adicionar informação de profit se houver
    if profit_usd != 0.0:
        if 'trading_logs' in st.session_state and st.session_state.trading_logs:
            st.session_state.trading_logs[-1]['profit_usd'] = profit_usd
            st.session_state.trading_logs[-1]['profit_brl'] = profit_usd * 5.5
