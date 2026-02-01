import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

class BacktestEngine:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def run_dynamic(self, portfolio_history, df_prices_all):
        """
        ウォークフォワード分析（年次リバランス）
        """
        print(f"  [Backtest] Running Walk-Forward Analysis ({min(portfolio_history.keys())}-{max(portfolio_history.keys())})...")
        
        # 全期間のリターンを格納するリスト
        all_daily_returns = []
        
        # 年ごとにループ
        sorted_years = sorted(portfolio_history.keys())
        for year in sorted_years:
            tickers = portfolio_history[year]
            
            # その年の開始日と終了日
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            # その年の株価データを抽出
            # df_prices_allのインデックスはDatetimeIndexであることを前提
            try:
                yearly_prices = df_prices_all.loc[start_date:end_date]
            except KeyError:
                continue
                
            if yearly_prices.empty:
                continue
                
            # 選ばれた銘柄だけのデータにする
            # 存在しない銘柄が含まれていた場合のハンドリング
            valid_tickers = [t for t in tickers if t in yearly_prices.columns]
            if not valid_tickers:
                # 銘柄がない場合は現金保有(リターン0)とする
                zeros = pd.Series(0, index=yearly_prices.index)
                all_daily_returns.append(zeros)
                continue
                
            yearly_prices = yearly_prices[valid_tickers]
            
            # 日次リターンを計算 (pct_change)
            returns = yearly_prices.pct_change().fillna(0)
            
            # 等ウェイトポートフォリオとして平均リターンを計算
            # axis=1 (横方向=銘柄方向) の平均
            portfolio_daily_ret = returns.mean(axis=1)
            
            all_daily_returns.append(portfolio_daily_ret)
            
        # 全期間を結合
        if not all_daily_returns:
            print("  [Error] No return data generated.")
            return None
            
        full_returns = pd.concat(all_daily_returns)
        
        # 重複日時の削除
        full_returns = full_returns[~full_returns.index.duplicated(keep='first')]
        
        # --- パフォーマンス指標の計算 ---
        # 累積リターン
        cumulative_returns = (1 + full_returns).cumprod()
        
        # S&P500 (ベンチマーク) の取得
        start_dt = full_returns.index[0]
        end_dt = full_returns.index[-1]
        try:
            sp500 = yf.download("^GSPC", start=start_dt, end=end_dt, progress=False)['Close']
            # MultiIndex対応
            if isinstance(sp500, pd.DataFrame):
                 sp500 = sp500.iloc[:, 0]
            sp500_returns = sp500.pct_change().fillna(0)
            sp500_cum = (1 + sp500_returns).cumprod()
            
            # インデックスを合わせる
            sp500_cum = sp500_cum.reindex(cumulative_returns.index, method='ffill')
        except:
            sp500_cum = None

        # CAGR
        days = (end_dt - start_dt).days
        total_return = cumulative_returns.iloc[-1]
        cagr = (total_return) ** (365.0 / days) - 1
        
        # Max Drawdown
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (Risk Free Rate = 0 assume)
        sharpe = full_returns.mean() / full_returns.std() * np.sqrt(252)

        return {
            'metrics': {
                'CAGR': cagr,
                'MaxDrawdown': max_drawdown,
                'Sharpe': sharpe
            },
            'data': {
                'cumulative_returns': cumulative_returns,
                'drawdown': drawdown,
                'daily_returns': full_returns,
                'benchmark': sp500_cum
            }
        }