import pandas as pd
from pattern_recognition import train_cnn, pattern_predict

# Dados simulados
data = pd.DataFrame({
    'open': [100, 102, 101, 105, 103],
    'close': [102, 101, 105, 103, 106]
})
model = train_cnn(data)
print(pattern_predict(data, model))
