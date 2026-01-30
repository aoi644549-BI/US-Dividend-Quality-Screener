import pandas as pd
import numpy as np
import yfinance as yf


class BacktestEngine:
    def __init__(self, start_date="2020-01-01", end_date="2024-01-01"):
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_ticker = "^GSPC"

    def run(self, tickers):
        print(f"バックテスト実行中: {len(tickers)} 銘柄...")
        
        # データ取得
        download_list = tickers + [self.benchmark_ticker]
        data = yf.download(download_list, start=self.start_date, end=self.end_date, auto_adjust=True)
        
        if 'Close' in data.columns.levels[0]:
            prices = data['Close']
        else:
            prices = data['Close']
            
        # リターン計算
        returns = prices.pct_change().dropna()
        
        # ポートフォリオ（均等加重）とベンチマークのリターン
        port_ret = returns[tickers].mean(axis=1)
        bench_ret = returns[self.benchmark_ticker]
        
        # 資産推移 (Cumulative Return)
        port_cum = (1 + port_ret).cumprod()
        bench_cum = (1 + bench_ret).cumprod()
        
        # 指標計算
        metrics = self._calculate_metrics(port_ret)
        
        return {
            'portfolio_cum': port_cum,
            'benchmark_cum': bench_cum,
            'metrics': metrics
        }

    def _calculate_metrics(self, returns):
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else 0
        
        cum = (1 + returns).cumprod()
        max_dd = ((cum / cum.cummax()) - 1).min()
        
        return {
            'CAGR': ann_return,
            'Sharpe': sharpe,
            'MaxDrawdown': max_dd
        }