import matplotlib.pyplot as plt
import os


class Plotter:
    def __init__(self, output_dir="output/figures"):
        # プロジェクトルートからのパスを解決
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.output_dir = os.path.join(project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_backtest(self, result, filename="backtest_result.png"):
        plt.figure(figsize=(12, 6))
        
        # ベンチマーク
        plt.plot(result['benchmark_cum'], label='S&P 500', color='gray', linestyle='--', alpha=0.7)
        # ポートフォリオ
        plt.plot(result['portfolio_cum'], label='My Strategy', color='#1f77b4', linewidth=2)
        
        # コロナショック
        plt.axvspan('2020-02-19', '2020-03-23', color='red', alpha=0.1, label='Covid-19 Crash')
        
        plt.title('Backtest Performance', fontsize=14)
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"グラフ保存完了: {save_path}")