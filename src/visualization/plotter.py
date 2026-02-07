import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.ticker as mtick
from lifelines import KaplanMeierFitter


class Plotter:
    def __init__(self, output_dir="output/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.rcParams['figure.figsize'] = [12, 6]
        sns.set_style("whitegrid")
        # フォントサイズを少し大きくして見やすく
        plt.rcParams.update({'font.size': 12})

    # === A/Bテスト比較用のプロット関数 ===
    def plot_ab_test_results(self, result_ai, result_simple):
        """AI戦略と単純戦略の比較グラフを作成する"""
        print("📊 A/Bテスト比較グラフの作成を開始します...")
        self.plot_ab_cumulative(result_ai, result_simple)
        self.plot_ab_drawdown(result_ai, result_simple)

    def plot_ab_cumulative(self, result_ai, result_simple):
        data_ai = result_ai['data']
        metrics_ai = result_ai['metrics']
        data_sim = result_simple['data']
        metrics_sim = result_simple['metrics']

        plt.figure(figsize=(12, 7))
        plt.plot(data_ai['cumulative_returns'], 
                 label=f"AI Strategy (CAGR: {metrics_ai['CAGR']:.2%}, Sharpe: {metrics_ai['Sharpe']:.2f})", 
                 linewidth=3, color='#2ca02c')
        plt.plot(data_sim['cumulative_returns'], 
                 label=f"Simple Yield (CAGR: {metrics_sim['CAGR']:.2%}, Sharpe: {metrics_sim['Sharpe']:.2f})", 
                 linewidth=2, color='#d62728', linestyle='--')
        
        bench = data_ai.get('benchmark', data_ai.get('benchmark_cum'))
        if bench is not None:
            days = (bench.index[-1] - bench.index[0]).days
            total_ret = (bench.iloc[-1] / bench.iloc[0])
            bench_cagr = total_ret ** (365.0 / days) - 1
            plt.plot(bench, label=f'S&P 500 (CAGR: {bench_cagr:.2%})', 
                     color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

        plt.title('A/B Test: Cumulative Return Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Growth of $1', fontsize=12)
        plt.xlabel('Year', fontsize=12)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "0_ab_test_cumulative.png"), dpi=300)
        plt.close()

    def plot_ab_drawdown(self, result_ai, result_simple):
        dd_ai = result_ai['data']['drawdown']
        dd_sim = result_simple['data']['drawdown']
        m_ai = result_ai['metrics']
        m_sim = result_simple['metrics']

        plt.figure(figsize=(12, 6))
        plt.plot(dd_sim, label=f"Simple Yield (Max DD: {m_sim['MaxDrawdown']:.2%})", 
                 color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.fill_between(dd_sim.index, dd_sim, 0, color='#d62728', alpha=0.1)
        plt.plot(dd_ai, label=f"AI Strategy (Max DD: {m_ai['MaxDrawdown']:.2%})", 
                 color='#2ca02c', linewidth=2)
        plt.fill_between(dd_ai.index, dd_ai, 0, color='#2ca02c', alpha=0.2)

        plt.title('A/B Test: Drawdown Comparison (Risk Management)', fontsize=16, fontweight='bold')
        plt.ylabel('Percentage from Peak', fontsize=12)
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=min(dd_sim.min(), dd_ai.min())*1.1, top=0.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "0_ab_test_drawdown.png"), dpi=300)
        plt.close()

    def plot_all(self, result, df_selected=None, survival_data=None, model_summary=None):
        """AI戦略単体の詳細グラフを描画する"""
        print("📊 AI戦略の詳細グラフの作成を開始します...")
        self.plot_cumulative_return(result)
        self.plot_drawdown(result)
        self.plot_annual_returns(result)
        self.plot_rolling_beta(result)
        self.plot_rolling_sharpe(result)
        self.plot_risk_return_scatter(result)
        self.plot_monthly_win_rate(result)
        
        if survival_data is not None and not survival_data.empty:
            self.plot_survival_curve(survival_data)
        if model_summary is not None and not model_summary.empty:
            self.plot_cox_coefficients(model_summary)
            
        print(f"✅ 全ての画像を {self.output_dir} に保存しました。")

    def plot_portfolio_yield(self, history, df_fund):
        """ポートフォリオの平均配当利回りの推移を計算してプロット"""
        years = []
        yields = []
        
        for year, tickers in history.items():
            subset = df_fund[df_fund['Symbol'].isin(tickers)]
            # 平均利回りを計算 (データがない場合は0)
            avg_yield = subset['DividendYield'].mean() if not subset.empty else 0
            years.append(year)
            yields.append(avg_yield)
            
        df_yield = pd.DataFrame({'Year': years, 'Yield': yields})
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_yield['Year'], df_yield['Yield'], color='#2ca02c', alpha=0.8)
        
        plt.title('Portfolio Average Dividend Yield History', fontsize=14, fontweight='bold')
        plt.ylabel('Average Dividend Yield')
        plt.xlabel('Year')
        
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2%}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
                     
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(df_yield['Year']) # 年ごとの目盛りを強制
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "14_portfolio_yield.png"), dpi=300)
        plt.close()

    def plot_cumulative_return(self, result):
        data = result['data']
        metrics = result['metrics']
        plt.figure(figsize=(12, 6))
        plt.plot(data['cumulative_returns'], label=f"AI Strategy (CAGR: {metrics['CAGR']:.2%})", linewidth=2, color='#2ca02c')
        bench = data.get('benchmark', data.get('benchmark_cum'))
        if bench is not None:
            plt.plot(bench, label='S&P 500', color='gray', linestyle='--', alpha=0.7)
        plt.title('Cumulative Return: AI Strategy vs S&P 500')
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
        plt.title('Drawdown Over Time')
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
        ai_annual = daily_ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        df_plot = pd.DataFrame({'AI Strategy': ai_annual})
        if bench_ret is not None:
            bench_annual = bench_ret.reindex(daily_ret.index).fillna(0).resample('YE').apply(lambda x: (1 + x).prod() - 1)
            df_plot['S&P 500'] = bench_annual
        df_plot.index = df_plot.index.year
        plt.figure(figsize=(12, 6))
        df_plot.plot(kind='bar', width=0.8, color=['#2ca02c', 'gray'], alpha=0.8)
        plt.title('Annual Returns Comparison')
        plt.ylabel('Annual Return')
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
        plt.title('Rolling Portfolio Beta')
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
        plt.title('Rolling Sharpe Ratio')
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
        plt.scatter(ai_vol, ai_cagr, color='#2ca02c', s=200, label='AI Strategy')
        plt.scatter(bench_vol, bench_cagr, color='gray', s=150, label='S&P 500')
        plt.title('Risk-Return Trade-off')
        plt.xlabel('Risk (Volatility)')
        plt.ylabel('Return (CAGR)')
        plt.savefig(os.path.join(self.output_dir, "11_risk_return_scatter.png"))
        plt.close()

    def plot_monthly_win_rate(self, result):
        if 'benchmark' not in result['data'] or result['data']['benchmark'] is None: return
        ai_ret = result['data']['daily_returns']
        bench_prices = result['data']['benchmark']
        bench_ret = bench_prices.pct_change().fillna(0).reindex(ai_ret.index).fillna(0)
        ai_monthly = ai_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        bench_monthly = bench_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        wins = ai_monthly > bench_monthly
        win_rate = wins.mean()
        plt.figure(figsize=(8, 6))
        plt.pie([win_rate, 1-win_rate], labels=['Win', 'Loss'], colors=['#2ca02c', '#d62728'], autopct='%1.1f%%', startangle=90)
        plt.title('Monthly Win Rate')
        plt.savefig(os.path.join(self.output_dir, "12_monthly_win_rate.png"))
        plt.close()

    def plot_survival_curve(self, df_survival):
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 6))
        kmf.fit(df_survival['T'], event_observed=df_survival['E'])
        kmf.plot_survival_function(color="#1f77b4", linewidth=2)
        plt.title('Kaplan-Meier Survival Curve (Dividend Sustainability)')
        plt.xlabel('Time (Financial Runway Proxy)')
        plt.ylabel('Survival Probability')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "9_survival_curve.png"))
        plt.close()

    def plot_cox_coefficients(self, summary_df):
        plt.figure(figsize=(10, 6))
        summary_sorted = summary_df.sort_values('coef', ascending=True)
        colors = ['#1f77b4' if c < 0 else '#d62728' for c in summary_sorted['coef']]
        plt.barh(summary_sorted.index, summary_sorted['coef'], color=colors)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.title('Key Risk Factors (Cox Model Coefficients)', fontsize=14, fontweight='bold')
        plt.xlabel('Coefficient (Log Hazard Ratio)')
        plt.ylabel('Financial Metrics')
        plt.text(0.05, 0.95, '← Safe (Increases Survival)', transform=plt.gca().transAxes, color='#1f77b4', fontsize=10)
        plt.text(0.70, 0.95, 'Risk (Increases Cut) →', transform=plt.gca().transAxes, color='#d62728', fontsize=10)
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "13_cox_coefficients.png"))
        plt.close()