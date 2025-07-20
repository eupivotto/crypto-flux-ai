import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import pandas as pd
import numpy as np

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Cálculo correto: 64x64 -> conv(3x3,pad=1) -> 64x64 -> maxpool(2x2) -> 32x32
        # 32 canais × 32 altura × 32 largura = 32.768
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten dinâmico
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.softmax(self.fc2(x), dim=1)

def generate_candle_image(data):
    try:
        if len(data) < 5:
            # Dados insuficientes, gerar imagem dummy
            img = Image.new('RGB', (64, 64), color='black')
            transform = transforms.Compose([
                transforms.Grayscale(), 
                transforms.Resize((64, 64)), 
                transforms.ToTensor()
            ])
            return transform(img).unsqueeze(0)
        
        # Usar apenas últimos 20 candles para performance
        data = data.tail(20).copy()
        
        plt.style.use('dark_background')  # Fundo escuro
        fig, ax = plt.subplots(figsize=(4, 4), dpi=16)  # 64x64 pixels
        
        for i, (idx, row) in enumerate(data.iterrows()):
            open_val = row['open']
            close_val = row['close']
            high_val = row.get('high', close_val + abs(close_val * 0.01))
            low_val = row.get('low', open_val - abs(open_val * 0.01))
            
            # Linha high-low
            ax.plot([i, i], [low_val, high_val], color='white', linewidth=1)
            
            # Corpo da vela
            color = 'green' if close_val > open_val else 'red'
            height = abs(close_val - open_val)
            bottom = min(open_val, close_val)
            
            rect = plt.Rectangle((i - 0.3, bottom), 0.6, height, 
                               facecolor=color, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)
        
        ax.set_xlim(-0.5, len(data) - 0.5)
        ax.set_ylim(data[['open', 'close']].min().min() * 0.99, 
                   data[['open', 'close']].max().max() * 1.01)
        ax.axis('off')
        
        # Converter para imagem
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = Image.fromarray(buf[:, :, :3])  # RGB apenas
        plt.close(fig)
        
        # Transformar para tensor
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        return transform(img).unsqueeze(0)
        
    except Exception as e:
        print(f"Erro ao gerar imagem: {e}")
        # Fallback: imagem preta
        img = Image.new('RGB', (64, 64), color='black')
        transform = transforms.Compose([
            transforms.Grayscale(), 
            transforms.Resize((64, 64)), 
            transforms.ToTensor()
        ])
        return transform(img).unsqueeze(0)

def pattern_predict(data, model):
    try:
        img = generate_candle_image(data)
        model.eval()
        with torch.no_grad():
            output = model(img)
            prediction = output.argmax(dim=1).item()
            confidence = output.max().item()
            
            # Retorna True se prediz alta E com confiança > 60%
            return prediction == 1 and confidence > 0.6
            
    except Exception as e:
        print(f"Erro em pattern_predict: {e}")
        return False

def train_cnn(data, epochs=5):
    try:
        if len(data) < 10:
            raise ValueError("Dados insuficientes para treinamento CNN.")
        
        model = CNNModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Treinar com múltiplas janelas de dados
        X, y = [], []
        window_size = 20
        
        for i in range(window_size, len(data)):
            window_data = data.iloc[i-window_size:i]
            img = generate_candle_image(window_data)
            X.append(img.squeeze(0))
            
            # Label baseado na tendência
            current_close = data.iloc[i]['close']
            prev_close = data.iloc[i-5]['close'] if i >= 5 else data.iloc[i-1]['close']
            y.append(1 if current_close > prev_close else 0)
        
        if len(X) < 5:
            raise ValueError("Poucos exemplos de treinamento gerados.")
        
        X = torch.stack(X)
        y = torch.tensor(y, dtype=torch.long)
        
        # Treinar
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for i in range(len(X)):
                optimizer.zero_grad()
                output = model(X[i:i+1])
                loss = criterion(output, y[i:i+1])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
        
        torch.save(model.state_dict(), 'cnn_model.pth')
        return model
        
    except Exception as e:
        print(f"Erro no treinamento CNN: {e}")
        raise
