import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out[:, -1, :]))

def calculate_indicators(data):
    if len(data) < 30:
        raise ValueError("Dados insuficientes para calcular indicadores. Precisa de pelo menos 30 linhas.")
    
    # Criar cópia para evitar modificar o original
    data = data.copy()
    
    data['EMA'] = data['close'].ewm(span=12, adjust=False).mean()
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Evitar divisão por zero
    rs = gain / loss.replace(0, 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    data['MACD'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    data['Bollinger_Mid'] = data['close'].rolling(20).mean()
    data['Bollinger_Std'] = data['close'].rolling(20).std()
    
    return data.dropna()

def normalize_features(features):
    """Normaliza features para estabilizar o treinamento"""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8  # Evita divisão por zero
    return (features - mean) / std

def technical_predict(data, model):
    try:
        indicators = calculate_indicators(data)
        if len(indicators) < 10:  # Precisa de sequência mínima
            return False
        
        # Pegar últimos 10 pontos para formar sequência temporal
        features = indicators[['EMA', 'RSI', 'MACD', 'Bollinger_Mid', 'Bollinger_Std']].values[-10:]
        features = normalize_features(features)
        
        # Formato correto: (batch_size=1, sequence_length=10, features=5)
        input_data = torch.tensor(features).float().unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            prediction = model(input_data)
            return prediction.item() > 0.5
            
    except Exception as e:
        print(f"Erro em technical_predict: {e}")
        return False

def train_lstm(data, epochs=10):
    try:
        indicators = calculate_indicators(data)
        if len(indicators) < 20:
            raise ValueError("Sequência muito curta para treinamento.")
        
        # Preparar dados em sequências
        sequence_length = 10
        features = indicators[['EMA', 'RSI', 'MACD', 'Bollinger_Mid', 'Bollinger_Std']].values
        features = normalize_features(features)
        
        # Criar sequências
        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(1.0 if indicators['close'].iloc[i] > indicators['close'].iloc[i-1] else 0.0)
        
        if len(X) == 0:
            raise ValueError("Não foi possível criar sequências de treinamento.")
        
        # Converter para tensors
        X = torch.tensor(np.array(X)).float()
        y = torch.tensor(y).float().unsqueeze(1)
        
        # Treinar modelo
        model = LSTMModel()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
        
        torch.save(model.state_dict(), 'lstm_model.pth')
        return model
        
    except Exception as e:
        print(f"Erro no treinamento LSTM: {e}")
        raise
