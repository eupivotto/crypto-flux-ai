from backtesting import Backtest, Strategy  # Import corrigido da biblioteca instalada
import pandas as pd

class MultiStrategy(Strategy):
    def init(self):
        # Inicialize indicadores ou ensemble aqui (ex: chame funções de outros módulos)
        pass

    def next(self):
        # Lógica de trade baseada em ensemble (ex: use predictions de IAs)
        if self.data.Close[-1] > self.data.Open[-1]:  # Exemplo simples; integre ensemble real
            self.buy()
        else:
            self.sell()

def run_backtest(data, strategy_class, initial_cash=1000, commission=0.002):
    # Converte dados para formato compatível (OHLCV)
    data = data.rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    data = data.set_index('Date')
    bt = Backtest(data, strategy_class, cash=initial_cash, commission=commission)
    return bt.run()  # Retorna resultados (ex: retorno, drawdown)
