import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Ajuste baseado em imagem 64x64 (exemplo)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: padrão de alta (1) ou baixa (0)
        self.dropout = nn.Dropout(0.2)  # Para evitar overfitting

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 32 * 32)  # Ajuste para o tamanho da imagem
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.softmax(self.fc2(x), dim=1)

def generate_candle_image(data):
    # Gera uma imagem simples de candles para teste (use Plotly para visual real no dashboard)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(data)):
        open_val = data['open'].iloc[i]
        close_val = data['close'].iloc[i]
        high_val = data.get('high', pd.Series([close_val + 1] * len(data))).iloc[i]
        low_val = data.get('low', pd.Series([open_val - 1] * len(data))).iloc[i]
        color = 'green' if close_val > open_val else 'red'
        ax.plot([i, i], [low_val, high_val], color='black')
        ax.add_patch(plt.Rectangle((i - 0.4, min(open_val, close_val)), 0.8, abs(close_val - open_val), color=color))
    ax.axis('off')
    fig.canvas.draw()
    # Correção: Use buffer_rgba e NumPy para extrair RGB de forma compatível
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img_array = buf[:, :, :3]  # Ignora o canal alpha para RGB
    img = Image.fromarray(img_array)
    plt.close(fig)
    transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((64, 64)), transforms.ToTensor()])
    return transform(img).unsqueeze(0)

def pattern_predict(data, model):
    img = generate_candle_image(data)
    model.eval()
    with torch.no_grad():
        output = model(img)
        return output.argmax().item() == 1  # True para padrão de alta (compra)

# Função para treinamento básico (use dados reais para produção)
def train_cnn(data, epochs=5):
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Simula labels: 1 para alta, 0 para baixa (baseado em close > open)
    labels = torch.tensor([1 if row['close'] > row['open'] else 0 for _, row in data.iterrows()]).long()
    for epoch in range(epochs):
        img = generate_candle_image(data)
        outputs = model(img)
        loss = criterion(outputs, labels[:1])  # Simples para teste; expanda para batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    torch.save(model.state_dict(), 'cnn_model.pth')
    return model
