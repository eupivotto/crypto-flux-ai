"""
Sistema de persistência de dados do AI Trading Bot
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class DataPersistence:
    """Classe para gerenciar persistência de dados"""
    
    def __init__(self):
        self.data_dir = "data"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Garante que o diretório de dados existe"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info("Diretório de dados criado")
    
    def save_trades_history(self, trades: List[Dict]) -> bool:
        """Salva histórico de trades"""
        try:
            filename = f"{self.data_dir}/trades_history_{datetime.now().strftime('%Y-%m')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(trades, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Histórico de trades salvo: {len(trades)} trades")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar histórico de trades: {e}")
            return False
    
    def load_trades_history(self) -> List[Dict]:
        """Carrega histórico de trades"""
        try:
            filename = f"{self.data_dir}/trades_history_{datetime.now().strftime('%Y-%m')}.json"
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
                logger.info(f"Histórico de trades carregado: {len(trades)} trades")
                return trades
        except Exception as e:
            logger.error(f"Erro ao carregar histórico de trades: {e}")
        return []
    
    def save_positions(self, open_positions: List[Dict], closed_positions: List[Dict]) -> bool:
        """Salva posições abertas e fechadas"""
        try:
            filename = f"{self.data_dir}/positions_{datetime.now().strftime('%Y-%m-%d')}.json"
            data = {
                'timestamp': datetime.now().isoformat(),
                'open_positions': open_positions,
                'closed_positions': closed_positions
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Posições salvas: {len(open_positions)} abertas, {len(closed_positions)} fechadas")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar posições: {e}")
            return False
    
    def save_performance_metrics(self, metrics: Dict) -> bool:
        """Salva métricas de performance"""
        try:
            filename = f"{self.data_dir}/performance_{datetime.now().strftime('%Y-%m-%d')}.json"
            metrics['timestamp'] = datetime.now().isoformat()
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
            logger.info("Métricas de performance salvas")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
            return False

# Instância global
persistence = DataPersistence()

def save_trade_to_history(trade_data: Dict) -> bool:
    """Função de compatibilidade para salvar trade individual"""
    try:
        # Carregar histórico existente
        trades = persistence.load_trades_history()
        
        # Adicionar novo trade
        trade_data['saved_at'] = datetime.now().isoformat()
        trades.append(trade_data)
        
        # Salvar histórico atualizado
        return persistence.save_trades_history(trades)
    except Exception as e:
        logger.error(f"Erro ao salvar trade individual: {e}")
        return False
