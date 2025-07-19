import pandas as pd
from custom_backtesting import run_backtest, MultiStrategy

# Dados simulados (substitua por get_data() do bot)
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-01-01', periods=5),
    'open': [100, 102, 101, 105, 103],
    'high': [102, 103, 102, 106, 104],
    'low': [99, 101, 100, 104, 102],
    'close': [101, 102, 101, 105, 103],
    'volume': [1000, 1100, 1200, 1300, 1400]
})
results = run_backtest(data, MultiStrategy)
print(results)  # Exibe métricas como retorno, drawdown
