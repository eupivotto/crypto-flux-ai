"""
Gerenciador de posições de trading
"""
from typing import List, Optional
import streamlit as st
from datetime import datetime
import time

from core.data_models import Position
from config.exchange_config import exchange
from config.settings import config
from utils.logger import get_logger
from utils.persistence import save_trade_to_history

logger = get_logger(__name__)

class PositionManager:
    """Gerenciador avançado de posições"""
    
    def __init__(self):
        self.monitoring_active = False
    
    def create_position(self, symbol: str, amount: float, entry_price: float, mode: str = 'Simulado') -> Optional[Position]:
        """Cria uma nova posição"""
        try:
            position = Position(symbol, entry_price, amount, mode)
            
            # Executar ordem de compra
            if mode == 'Testnet':
                success = self._execute_buy_order(position)
                if not success:
                    return None
            else:
                logger.info(f"Ordem simulada: {symbol} @ ${entry_price:.2f}")
            
            # Adicionar às posições abertas
            if 'open_positions' not in st.session_state:
                st.session_state.open_positions = []
            
            st.session_state.open_positions.append(position.to_dict())
            
            if 'position_counter' not in st.session_state:
                st.session_state.position_counter = 0
            st.session_state.position_counter += 1
            
            logger.info(f"Nova posição criada: {position.id}")
            return position
            
        except Exception as e:
            logger.error(f"Erro ao criar posição: {e}")
            return None
    
    def _execute_buy_order(self, position: Position) -> bool:
        """Executa ordem de compra real"""
        try:
            buy_order = exchange.create_market_buy_order(position.symbol, position.amount)
            position.buy_order_id = buy_order['id']
            position.entry_price = buy_order.get('average', position.entry_price)
            
            logger.info(f"Ordem real executada: {buy_order['id'][:8]}")
            return True
            
        except Exception as e:
            logger.error(f"Erro na ordem real: {e}")
            return False
    
    def monitor_positions(self):
        """Monitora todas as posições abertas"""
        if not getattr(st.session_state, 'auto_sell_enabled', True):
            return
        
        if 'open_positions' not in st.session_state:
            st.session_state.open_positions = []
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
                
                # Verificar saída por tempo (15 minutos para scalping)
                entry_time = datetime.fromisoformat(pos_dict['entry_time'])
                duration_minutes = (datetime.now() - entry_time).total_seconds() / 60
                
                if duration_minutes > 15:  # 15 minutos máximo
                    pnl_percent = (current_price - entry_price) / entry_price
                    if pnl_percent > 0.001:  # Só sair por tempo se tiver pelo menos 0.1% de lucro
                        self.close_position(position_id, current_price, 'TIME_EXIT')
                        positions_to_close.append(position_id)
            
            # Atualizar timestamp do último check
            st.session_state.last_position_check = time.time()
            
            if positions_to_close:
                logger.info(f"{len(positions_to_close)} posições fechadas automaticamente")
                
        except Exception as e:
            logger.error(f"Erro no monitoramento: {e}")
    
    def close_position(self, position_id: str, exit_price: float, exit_reason: str) -> bool:
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
                logger.error(f"Posição {position_id} não encontrada")
                return False
            
            # Executar ordem de venda
            if position_dict['mode'] == 'Testnet':
                success = self._execute_sell_order(position_dict, exit_price)
                if not success:
                    return False
            else:
                logger.info(f"Venda simulada: {position_dict['symbol']} @ ${exit_price:.2f}")
            
            # Calcular P&L
            entry_price = position_dict['entry_price']
            amount = position_dict['amount']
            pnl_usd = (exit_price - entry_price) * amount
            pnl_brl = pnl_usd * config.USD_TO_BRL
            
            # Atualizar dados da posição
            position_dict.update({
                'status': 'CLOSED',
                'exit_price': exit_price,
                'exit_time': datetime.now().isoformat(),
                'exit_reason': exit_reason,
                'pnl_usd': pnl_usd,
                'pnl_brl': pnl_brl
            })
            
            # Mover para posições fechadas
            if 'closed_positions' not in st.session_state:
                st.session_state.closed_positions = []
            
            st.session_state.closed_positions.append(position_dict)
            del st.session_state.open_positions[position_index]
            
            # Atualizar lucros globais
            if 'daily_profit_usd' not in st.session_state:
                st.session_state.daily_profit_usd = 0.0
            if 'daily_profit' not in st.session_state:
                st.session_state.daily_profit = 0.0
            
            st.session_state.daily_profit_usd += pnl_usd
            st.session_state.daily_profit += pnl_brl
            
            # Registrar no histórico de trades
            trade_record = {
                'id': len(getattr(st.session_state, 'trade_history', [])) + 1,
                'timestamp_str': datetime.now().strftime("%H:%M:%S"),
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
                'duration_minutes': (datetime.now() - datetime.fromisoformat(position_dict['entry_time'])).total_seconds() / 60
            }
            
            if 'trade_history' not in st.session_state:
                st.session_state.trade_history = []
            st.session_state.trade_history.append(trade_record)
            save_trade_to_history(trade_record)
            
            result_type = "LUCRO" if pnl_usd > 0 else "PREJUÍZO"
            logger.info(f"{exit_reason}: {result_type} ${pnl_usd:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {e}")
            return False
    
    def _execute_sell_order(self, position_dict: dict, exit_price: float) -> bool:
        """Executa ordem de venda real"""
        try:
            sell_order = exchange.create_market_sell_order(
                position_dict['symbol'], 
                position_dict['amount']
            )
            exit_price = sell_order.get('average', exit_price)
            position_dict['sell_order_id'] = sell_order['id']
            
            logger.info(f"Venda real executada: {sell_order['id'][:8]}")
            return True
            
        except Exception as e:
            logger.error(f"Erro na venda real: {e}")
            return False
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual de um símbolo"""
        try:
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Erro ao obter preço de {symbol}: {e}")
            return None
    
    def get_open_positions_count(self) -> int:
        """Retorna número de posições abertas"""
        if 'open_positions' not in st.session_state:
            return 0
        return len([pos for pos in st.session_state.open_positions if pos['status'] == 'OPEN'])
    
    def get_unrealized_pnl(self) -> float:
        """Calcula P&L não realizado de todas as posições abertas"""
        if 'open_positions' not in st.session_state:
            return 0.0
        
        total_pnl = 0.0
        
        for pos in st.session_state.open_positions:
            if pos['status'] != 'OPEN':
                continue
                
            current_price = self.get_current_price(pos['symbol'])
            if current_price:
                pnl = (current_price - pos['entry_price']) * pos['amount']
                total_pnl += pnl
        
        return total_pnl
