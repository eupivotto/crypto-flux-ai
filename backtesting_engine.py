import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class BacktestEngine:
    def __init__(self, initial_capital=1000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def add_signal(self, timestamp, price, signal, confidence=0.0):
        """Adiciona sinal de trading ao backtest"""
        self.positions.append({
            'timestamp': timestamp,
            'price': price,
            'signal': signal,  # 1 para compra, -1 para venda, 0 para hold
            'confidence': confidence
        })
    
    def calculate_returns(self, data, signals):
        """Calcula retornos baseado nos sinais"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        
        for i, (idx, row) in enumerate(data.iterrows()):
            current_price = row['close']
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Lógica de entrada (compra)
            if signal == 1 and position == 0:
                position = (capital * 0.95) / current_price  # 95% do capital (reserva para taxas)
                entry_price = current_price
                capital -= position * current_price * (1 + self.commission)
                
                self.trades.append({
                    'entry_time': row['timestamp'],
                    'entry_price': entry_price,
                    'type': 'BUY',
                    'quantity': position
                })
            
            # Lógica de saída (venda)
            elif signal == -1 and position > 0:
                capital += position * current_price * (1 - self.commission)
                
                # Calcular P&L do trade
                pnl = (current_price - entry_price) * position - (entry_price * position * self.commission * 2)
                
                self.trades[-1].update({
                    'exit_time': row['timestamp'],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'return_pct': (current_price - entry_price) / entry_price * 100
                })
                
                position = 0
                entry_price = 0
            
            # Calcular valor atual do portfólio
            portfolio_value = capital + (position * current_price if position > 0 else 0)
            
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'portfolio_value': portfolio_value,
                'price': current_price,
                'position': position
            })
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calcula métricas de performance"""
        if not self.trades or not self.equity_curve:
            return {}
        
        # Converter para DataFrames
        trades_df = pd.DataFrame([t for t in self.trades if 'exit_time' in t])
        equity_df = pd.DataFrame(self.equity_curve)
        
        if trades_df.empty:
            return {'error': 'Nenhum trade completo encontrado'}
        
        # Métricas básicas
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # Retornos
        total_return = (equity_df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Win Rate
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Drawdown
        equity_df['running_max'] = equity_df['portfolio_value'].expanding().max()
        equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['running_max']) / equity_df['running_max'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe Ratio
        returns = equity_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': gross_profit / winning_trades if winning_trades > 0 else 0,
            'avg_loss': gross_loss / losing_trades if losing_trades > 0 else 0,
            'trades_df': trades_df,
            'equity_df': equity_df
        }
    
    def generate_signals_from_ai(self, data, ai_predictions):
        """Gera sinais baseado nas predições das IAs"""
        signals = []
        
        for i, predictions in enumerate(ai_predictions):
            if len(predictions) >= 4:  # [tech, pattern, sentiment, risk]
                # Usar mesma lógica do weighted_vote
                weights = [0.35, 0.20, 0.15, 0.30]
                confidence = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
                
                if confidence > 0.7 and predictions[3]:  # Confiança > 70% E risco OK
                    signals.append(1)  # Compra
                elif len(signals) > 0 and signals[-1] == 1:  # Se estava em posição
                    # Criar lógica de saída
                    if confidence < 0.3:  # Confiança baixa = venda
                        signals.append(-1)
                    else:
                        signals.append(0)  # Hold
                else:
                    signals.append(0)  # Hold
            else:
                signals.append(0)
        
        return pd.Series(signals)

def simulate_ai_predictions(data, num_predictions=None):
    """Simula predições das IAs para backtesting"""
    if num_predictions is None:
        num_predictions = len(data)
    
    predictions = []
    
    for i in range(num_predictions):
        # Simular análise técnica baseada em momentum
        if i > 5:
            price_momentum = data['close'].iloc[i] > data['close'].iloc[i-5]
        else:
            price_momentum = True
        
        # Simular reconhecimento de padrões (baseado em volatilidade)
        if i > 10:
            recent_vol = data['close'].iloc[i-10:i].std()
            pattern_signal = recent_vol < data['close'].iloc[i-20:i].std() if i > 20 else True
        else:
            pattern_signal = True
        
        # Simular sentimento (aleatório com tendência)
        sentiment = np.random.random() > 0.4
        
        # Simular gestão de risco (baseado em volatilidade)
        if i > 14:
            risk_ok = data['close'].iloc[i-14:i].std() / data['close'].iloc[i] < 0.05
        else:
            risk_ok = True
        
        predictions.append([price_momentum, pattern_signal, sentiment, risk_ok])
    
    return predictions
