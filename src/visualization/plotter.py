import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from lifelines import KaplanMeierFitter 


class Plotter:
    def __init__(self, output_dir="output/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.rcParams['figure.figsize'] = [12, 6]
        sns.set_style("whitegrid")

    def plot_all(self, result, df_selected=None, survival_data=None):
        """全てのグラフを描画する"""
        print("📊 グラフの作成を開始します...")
        
        # 1. パフォーマンス系
        self.plot_cumulative_return(result)
        self.plot_drawdown(result)
        self.plot_annual_returns(result)
        
        # 2. リスク指標系
        self.plot_rolling_beta(result)
        self.plot_rolling_sharpe(result)
        self.plot_risk_return_scatter(result)
        self.plot_monthly_win_rate(result)
        
        # 3. 生存曲線
        if survival_data is not None and not survival_data.empty:
            self.plot_survival_curve(survival_data)
        else:
            print("  [Info] 生存解析データがないため、生存曲線はスキップします。")

        print(f"✅ 全ての画像を {self.output_dir} に保存しました。")

    def plot_cumulative_return(self, result):
        data = result['data']
        metrics = result['metrics']
        plt.figure(figsize=(12, 6))
        cum_ret = data['cumulative_returns']
        plt.plot(cum_ret, label=f"AI Strategy (CAGR: {metrics['CAGR']:.2%})", linewidth=2, color='#2ca02c')
        bench = data.get('benchmark', data.get('benchmark_cum'))
        if bench is not None:
            plt.plot(bench, label='S&P 500', color='gray', linestyle='--', alpha=0.7)
        plt.title('Cumulative Return: AI Strategy vs S&P 500', fontsize=14)
        plt.ylabel('Growth of $1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "1_cumulative_return.png"))
        plt.close()

    def plot_drawdown(self, result):
        data = result['data']
        drawdown = data['drawdown']
        plt.figure(figsize=(12, 4))
        plt.plot(drawdown, color='#d62728', linewidth=1)
        plt.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
        plt.title('Drawdown Over Time', fontsize=14)
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "2_drawdown.png"))
        plt.close()

    def plot_annual_returns(self, result):
        daily_ret = result['data']['daily_returns']
        bench_ret = None
        if 'benchmark' in result['data'] and result['data']['benchmark'] is not None:
             bench_prices = result['data']['benchmark']
             bench_ret = bench_prices.pct_change().fillna(0)
        ai_annual = daily_ret.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        df_plot = pd.DataFrame({'AI Strategy': ai_annual})
        if bench_ret is not None:
            bench_annual = bench_ret.reindex(daily_ret.index).fillna(0).resample('Y').apply(lambda x: (1 + x).prod() - 1)
            df_plot['S&P 500'] = bench_annual
        df_plot.index = df_plot.index.year
        plt.figure(figsize=(12, 6))
        df_plot.plot(kind='bar', width=0.8, color=['#2ca02c', 'gray'], alpha=0.8)
        plt.title('Annual Returns Comparison')
        plt.ylabel('Annual Return')
        plt.grid(True, axis='y', alpha=0.3)
        plt.axhline(0, color='black', linewidth=1)
        plt.savefig(os.path.join(self.output_dir, "3_annual_returns.png"))
        plt.close()

    def plot_rolling_beta(self, result, window=252):
        if 'benchmark' not in result['data'] or result['data']['benchmark'] is None: return
        ai_ret = result['data']['daily_returns']
        bench_prices = result['data']['benchmark']
        bench_ret = bench_prices.pct_change().fillna(0).reindex(ai_ret.index).fillna(0)
        rolling_cov = ai_ret.rolling(window=window).cov(bench_ret)
        rolling_var = bench_ret.rolling(window=window).var()
        rolling_beta = rolling_cov / rolling_var
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_beta, label='Rolling Beta (1Y)', color='#1f77b4')
        plt.axhline(1.0, color='red', linestyle='--', label='Market Risk (Beta=1)')
        plt.title('Rolling Portfolio Beta (Risk Exposure)')
        plt.savefig(os.path.join(self.output_dir, "7_rolling_beta.png"))
        plt.close()

    def plot_rolling_sharpe(self, result, window=252):
        if 'benchmark' not in result['data'] or result['data']['benchmark'] is None: return
        ai_ret = result['data']['daily_returns']
        bench_prices = result['data']['benchmark']
        bench_ret = bench_prices.pct_change().fillna(0).reindex(ai_ret.index).fillna(0)
        ai_rolling_sharpe = ai_ret.rolling(window).mean() / ai_ret.rolling(window).std() * np.sqrt(252)
        bench_rolling_sharpe = bench_ret.rolling(window).mean() / bench_ret.rolling(window).std() * np.sqrt(252)
        plt.figure(figsize=(12, 6))
        plt.plot(ai_rolling_sharpe, label='AI Strategy', color='#2ca02c')
        plt.plot(bench_rolling_sharpe, label='S&P 500', color='gray', linestyle='--', alpha=0.7)
        plt.title('Rolling Sharpe Ratio (1-Year Window)')
        plt.savefig(os.path.join(self.output_dir, "10_rolling_sharpe.png"))
        plt.close()

    def plot_risk_return_scatter(self, result):
        if 'benchmark' not in result['data'] or result['data']['benchmark'] is None: return
        ai_ret = result['data']['daily_returns']
        bench_prices = result['data']['benchmark']
        bench_ret = bench_prices.pct_change().fillna(0).reindex(ai_ret.index).fillna(0)
        ai_cagr = result['metrics']['CAGR']
        ai_vol = ai_ret.std() * np.sqrt(252)
        days = (bench_prices.index[-1] - bench_prices.index[0]).days
        bench_total_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0])
        bench_cagr = bench_total_ret ** (365.0 / days) - 1
        bench_vol = bench_ret.std() * np.sqrt(252)
        plt.figure(figsize=(10, 8))
        plt.scatter(ai_vol, ai_cagr, color='#2ca02c', s=200, label='AI Strategy', zorder=5)
        plt.text(ai_vol, ai_cagr + 0.005, '  AI Model', fontsize=12, fontweight='bold')
        plt.scatter(bench_vol, bench_cagr, color='gray', s=150, label='S&P 500', zorder=5)
        plt.text(bench_vol, bench_cagr - 0.01, '  S&P 500', fontsize=12)
        plt.title('Risk-Return Trade-off (10 Years)')
        plt.xlabel('Annualized Risk (Volatility)')
        plt.ylabel('Annualized Return (CAGR)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.output_dir, "11_risk_return_scatter.png"))
        plt.close()

    def plot_monthly_win_rate(self, result):
        if 'benchmark' not in result['data'] or result['data']['benchmark'] is None: return
        ai_ret = result['data']['daily_returns']
        bench_prices = result['data']['benchmark']
        bench_ret = bench_prices.pct_change().fillna(0).reindex(ai_ret.index).fillna(0)
        ai_monthly = ai_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
        bench_monthly = bench_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
        wins = ai_monthly > bench_monthly
        win_rate = wins.mean()
        plt.figure(figsize=(8, 6))
        plt.pie([win_rate, 1-win_rate], labels=['Win', 'Loss'], colors=['#2ca02c', '#d62728'], autopct='%1.1f%%', startangle=90)
        plt.title(f'Monthly Win Rate vs S&P 500\n(Total Months: {len(ai_monthly)})')
        plt.savefig(os.path.join(self.output_dir, "12_monthly_win_rate.png"))
        plt.close()

    #  生存曲線 ---
    def plot_survival_curve(self, df_survival):
        """9. カプラン＝マイヤー曲線の描画"""
        kmf = KaplanMeierFitter()
        
        plt.figure(figsize=(10, 6))
        # データを使ってフィッティングし、プロット
        # E=1 (Event発生: 減配/低財務), T=Time
        kmf.fit(df_survival['T'], event_observed=df_survival['E'])
        kmf.plot_survival_function(color="#1f77b4")
        
        plt.title('Kaplan-Meier Survival Curve (Dividend Sustainability)', fontsize=14, fontweight='bold')
        plt.xlabel('Time (Financial Runway Proxy)')
        plt.ylabel('Survival Probability (No Dividend Cut)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "9_survival_curve.png"))
        plt.close()