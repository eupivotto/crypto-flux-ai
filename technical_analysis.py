import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)  # Adicionado para evitar overfitting
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out[:, -1, :]))

def calculate_indicators(data):
    if len(data) < 30:  # Verificação mínima para rolling windows
        raise ValueError("Dados insuficientes para calcular indicadores. Precisa de pelo menos 30 linhas.")
    data['EMA'] = data['close'].ewm(span=12, adjust=False).mean()
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + gain / loss))
    data['MACD'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    data['Bollinger_Mid'] = data['close'].rolling(20).mean()
    data['Bollinger_Std'] = data['close'].rolling(20).std()
    return data.dropna()

def technical_predict(data, model):
    indicators = calculate_indicators(data)
    if len(indicators) == 0:
        return False  # Default para nenhum sinal
    input_data = torch.tensor(indicators[['EMA', 'RSI', 'MACD', 'Bollinger_Mid', 'Bollinger_Std']].values).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        return model(input_data).item() > 0.5  # True para compra

# Função para treinamento (com divisão treino/teste)
def train_lstm(data, epochs=10):
    indicators = calculate_indicators(data)
    if len(indicators) < 2:
        raise ValueError("Sequência de dados muito curta após processamento.")
    
    # Divisão simples: 80% treino, 20% teste
    split = int(0.8 * len(indicators))
    train_data = indicators.iloc[:split]
    # Rótulos simples: 1 se close > open (alta), 0 otherwise
    train_inputs = torch.tensor(train_data[['EMA', 'RSI', 'MACD', 'Bollinger_Mid', 'Bollinger_Std']].values).float().unsqueeze(0)
    train_targets = torch.tensor([1.0 if data['close'].iloc[i] > data['open'].iloc[i] else 0.0 for i in range(split)]).float().unsqueeze(0).unsqueeze(2)
    
    model = LSTMModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets[:, -1, :])  # Use último da sequência para simplificar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), 'lstm_model.pth')
    return model
