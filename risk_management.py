import numpy as np
import pandas as pd

def calculate_var(returns, confidence=0.95):
    """Calcula Value at Risk (VaR) histórico."""
    sorted_returns = np.sort(returns)
    index = int((1 - confidence) * len(sorted_returns))
    return abs(sorted_returns[index])  # VaR absoluto

def calculate_volatility(returns):
    """Calcula volatilidade (desvio padrão)."""
    return returns.std()

def risk_assess(data, current_capital, var_threshold=0.05, vol_threshold=0.1):
    """Avalia risco geral: VaR, volatilidade e correlações simples."""
    returns = data['close'].pct_change().dropna()
    if len(returns) < 2:
        return True  # Default se dados insuficientes
    var = calculate_var(returns)
    volatility = calculate_volatility(returns)
    # Correlação simples com um ativo de referência (ex: BTC vs. mercado)
    correlation = returns.corr(returns.shift(1)) if len(returns) > 1 else 0
    # Decisão: Risco OK se abaixo dos thresholds
    if var > var_threshold or volatility > vol_threshold or abs(correlation) > 0.9:
        return False  # Risco alto, não trade
    # Ajuste baseado no capital (ex: perda potencial < 3% do capital)
    potential_loss = var * current_capital
    if potential_loss > 0.03 * current_capital:
        return False
    return True  # Risco aceitável
