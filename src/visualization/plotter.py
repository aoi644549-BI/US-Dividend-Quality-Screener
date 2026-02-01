%%writefile src/visualization/plotter.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from lifelines import KaplanMeierFitter 


class Plotter:
    def __init__(self, output_dir="output/figures"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.output_dir = os.path.join(project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'

    def plot_all(self, result, df_selected, survival_data=None): 
        """全てのグラフを一括生成する"""
        if result is None:
            return

        self.plot_cumulative_return(result)
        self.plot_drawdown(result)
        self.plot_rolling_sharpe(result)
        self.plot_sector_allocation(df_selected)
        self.plot_monthly_heatmap(result)
        self.plot_return_distribution(result)
        self.plot_rolling_beta(result)
        self.plot_monte_carlo(result)
        
        if survival_data is not None:
            self.plot_survival_curve(survival_data)
            
        print(f"📊 全てのグラフを {self.output_dir} に保存しました。")

    def plot_cumulative_return(self, result):
        plt.figure(figsize=(12, 6))
        plt.plot(result['benchmark_cum'], label='S&P 500', color='gray', linestyle='--', alpha=0.7)
        plt.plot(result['portfolio_cum'], label='AI Strategy', color='#1f77b4', linewidth=2)
        plt.title('Cumulative Return (10 Years)', fontsize=14, fontweight='bold')
        plt.ylabel('Growth of $1')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "1_cumulative_return.png"))
        plt.close()

    def plot_drawdown(self, result):
        port_cum = result['portfolio_cum']
        drawdown = (port_cum / port_cum.cummax()) - 1
        plt.figure(figsize=(12, 4))
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown, color='red', linewidth=1, label='Drawdown')
        plt.title('Underwater Plot (Drawdown Risk)', fontsize=14, fontweight='bold')
        plt.ylabel('Drawdown (%)')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "2_drawdown.png"))
        plt.close()

    def plot_rolling_sharpe(self, result, window=252):
        port_ret = result['portfolio_cum'].pct_change().dropna()
        def calc_sharpe(series):
            if series.std() == 0: return 0
            return (series.mean() * 252) / (series.std() * np.sqrt(252))
        rolling_sharpe = port_ret.rolling(window).apply(calc_sharpe)
        plt.figure(figsize=(12, 5))
        plt.plot(rolling_sharpe, label='Rolling Sharpe (1y)', color='green')
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='Good (>1.0)')
        plt.title('Rolling Sharpe Ratio (1-Year Window)', fontsize=14, fontweight='bold')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "3_rolling_sharpe.png"))
        plt.close()

    def plot_sector_allocation(self, df_selected):
        if 'Original_Sector' in df_selected.columns:
            counts = df_selected['Original_Sector'].value_counts()
        else:
            counts = df_selected['Sector'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        plt.title('Portfolio Sector Allocation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "4_sector_allocation.png"))
        plt.close()

    def plot_monthly_heatmap(self, result):
        port_ret = result['portfolio_cum'].pct_change().dropna()
        monthly_ret = port_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        df_ret = pd.DataFrame({'Year': monthly_ret.index.year, 'Month': monthly_ret.index.month, 'Return': monthly_ret.values})
        pivot_table = df_ret.pivot(index='Year', columns='Month', values='Return')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Return (%)'})
        plt.title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Year')
        plt.xlabel('Month')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "5_monthly_heatmap.png"))
        plt.close()

    def plot_return_distribution(self, result):
        port_ret = result['portfolio_cum'].pct_change().dropna()
        plt.figure(figsize=(10, 6))
        sns.histplot(port_ret, bins=50, kde=True, stat="density", label='Portfolio Returns', color='#1f77b4', alpha=0.6)
        mu, std = port_ret.mean(), port_ret.std()
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, linestyle='--', label='Normal Distribution')
        stats_text = f"Mean: {mu:.4f}\nStd: {std:.4f}\nSkew: {port_ret.skew():.2f}\nKurt: {port_ret.kurt():.2f}"
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title('Return Distribution Analysis (Fat Tail Check)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "6_return_distribution.png"))
        plt.close()

    def plot_rolling_beta(self, result, window=126):
        port_ret = result['portfolio_cum'].pct_change().dropna()
        bench_ret = result['benchmark_cum'].pct_change().dropna()
        rolling_cov = port_ret.rolling(window).cov(bench_ret)
        rolling_var = bench_ret.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var
        plt.figure(figsize=(12, 5))
        plt.plot(rolling_beta, color='purple', label=f'Rolling Beta ({window} days)')
        plt.axhline(1.0, color='gray', linestyle='--', label='Market (Beta=1)')
        plt.axhline(rolling_beta.mean(), color='orange', linestyle=':', label=f'Avg Beta ({rolling_beta.mean():.2f})')
        plt.title('Rolling Beta (Sensitivity to Market)', fontsize=14, fontweight='bold')
        plt.ylabel('Beta')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "7_rolling_beta.png"))
        plt.close()

    def plot_monte_carlo(self, result, n_simulations=1000, days=252*10):
        port_ret = result['portfolio_cum'].pct_change().dropna()
        mu = port_ret.mean()
        sigma = port_ret.std()
        start_price = result['portfolio_cum'].iloc[-1]
        simulations = np.zeros((days, n_simulations))
        for i in range(n_simulations):
            random_shocks = np.random.normal(mu, sigma, days)
            price_path = start_price * (1 + random_shocks).cumprod()
            simulations[:, i] = price_path
        plt.figure(figsize=(12, 6))
        plt.plot(simulations[:, :100], color='#1f77b4', alpha=0.05)
        mean_path = np.mean(simulations, axis=1)
        plt.plot(mean_path, color='orange', linewidth=2, label='Mean Scenario')
        plt.title(f'Monte Carlo Simulation ({n_simulations} Scenarios)', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "8_monte_carlo.png"))
        plt.close()

    def plot_survival_curve(self, df_survival):
        """9. カプラン＝マイヤー曲線の描画"""
        kmf = KaplanMeierFitter()
        
        plt.figure(figsize=(10, 6))
        # データを使ってフィッティングし、プロット
        kmf.fit(df_survival['T'], event_observed=df_survival['E'])
        kmf.plot_survival_function(color="#1f77b4")
        
        plt.title('Kaplan-Meier Survival Curve (Dividend Sustainability)', fontsize=14, fontweight='bold')
        plt.xlabel('Time (proxy based on Payout Ratio)')
        plt.ylabel('Survival Probability (No Dividend Cut)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "9_survival_curve.png"))
        plt.close()