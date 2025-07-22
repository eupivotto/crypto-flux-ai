import pandas as pd
import numpy as np
from pattern_recognition import train_cnn, pattern_predict, CNNModel

# Gerar dados sintéticos maiores para treinamento
np.random.seed(42)
n_samples = 50  # Pelo menos 50 amostras

# Simular dados OHLC realistas
base_price = 50000  # Preço base do BTC
price_data = []

for i in range(n_samples):
    # Simular movimento aleatório
    change = np.random.normal(0, 0.02)  # 2% de volatilidade
    
    if i == 0:
        open_price = base_price
    else:
        open_price = price_data[-1]['close']
    
    close_price = open_price * (1 + change)
    high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
    low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
    
    price_data.append({
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': np.random.uniform(1000, 5000)
    })

# Criar DataFrame
data = pd.DataFrame(price_data)
print(f"Dados gerados: {len(data)} linhas")
print(data.head())

# Treinar o modelo
try:
    model = train_cnn(data, epochs=5)
    print("✅ Modelo CNN treinado com sucesso!")
    
    # Testar predição
    prediction = pattern_predict(data.tail(20), model)
    print(f"Predição de padrão: {prediction}")
    
except Exception as e:
    print(f"❌ Erro: {e}")
