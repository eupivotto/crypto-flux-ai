import pandas as pd
import numpy as np
from technical_analysis import train_lstm, technical_predict

# Dados simulados mais longos (50 candles aleatórios para teste)
np.random.seed(42)  # Para reproducibilidade
closes = 100 + np.cumsum(np.random.randn(50) * 2)  # Simula preços com ruído
opens = closes - np.random.rand(50) * 1  # Opens ligeiramente menores
data = pd.DataFrame({'close': closes, 'open': opens})

try:
    model = train_lstm(data, epochs=5)  # Menos epochs para teste rápido
    prediction = technical_predict(data, model)
    print(f"Previsão de compra: {prediction}")
except ValueError as e:
    print(f"Erro nos dados: {e}")
