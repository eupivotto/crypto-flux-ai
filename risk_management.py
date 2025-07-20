import numpy as np
import pandas as pd

def calculate_var(returns, confidence=0.95):
    """Calcula Value at Risk (VaR) histórico."""
    try:
        if len(returns) < 2:
            return 0.01  # VaR padrão conservador
        
        sorted_returns = np.sort(returns.dropna())
        index = max(0, int((1 - confidence) * len(sorted_returns)) - 1)
        return abs(sorted_returns[index])
        
    except Exception as e:
        print(f"Erro ao calcular VaR: {e}")
        return 0.01

def calculate_volatility(returns):
    """Calcula volatilidade (desvio padrão)."""
    try:
        if len(returns) < 2:
            return 0.01
        return returns.std()
    except Exception as e:
        print(f"Erro ao calcular volatilidade: {e}")
        return 0.01

def risk_assess(data, current_capital, var_threshold=0.05, vol_threshold=0.1):
    """Avalia risco geral: VaR, volatilidade e correlações simples."""
    try:
        if len(data) < 10:
            return True  # Dados insuficientes, permitir trade
        
        returns = data['close'].pct_change().dropna()
        if len(returns) < 5:
            return True
        
        var = calculate_var(returns)
        volatility = calculate_volatility(returns)
        
        # Correlação autocorrelação com tratamento de NaN
        correlation = 0
        try:
            corr_result = returns.corr(returns.shift(1))
            if not pd.isna(corr_result):
                correlation = abs(corr_result)
        except:
            correlation = 0
        
        print(f"Risk Assessment - VaR: {var:.4f}, Vol: {volatility:.4f}, Corr: {correlation:.4f}")
        
        # Decisão de risco
        risk_factors = 0
        
        if var > var_threshold:
            risk_factors += 1
            print(f"⚠️ VaR alto: {var:.4f} > {var_threshold}")
        
        if volatility > vol_threshold:
            risk_factors += 1
            print(f"⚠️ Volatilidade alta: {volatility:.4f} > {vol_threshold}")
        
        if correlation > 0.9:
            risk_factors += 1
            print(f"⚠️ Correlação alta: {correlation:.4f}")
        
        # Verificar perda potencial
        potential_loss = var * current_capital
        max_loss_threshold = 0.03 * current_capital
        
        if potential_loss > max_loss_threshold:
            risk_factors += 1
            print(f"⚠️ Perda potencial: ${potential_loss:.2f} > ${max_loss_threshold:.2f}")
        
        # Decisão: permitir se risco baixo
        risk_ok = risk_factors <= 1  # Máximo 1 fator de risco
        
        print(f"✅ Risco {'ACEITÁVEL' if risk_ok else 'ALTO'} - Fatores: {risk_factors}/4")
        
        return risk_ok
        
    except Exception as e:
        print(f"Erro em risk_assess: {e}")
        return True  # Default: permitir trade se erro
