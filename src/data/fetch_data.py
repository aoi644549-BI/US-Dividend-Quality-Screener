import pandas as pd
import yfinance as yf
import requests
import io
import time
import os
from tqdm import tqdm
import warnings


warnings.simplefilter('ignore')

def fetch_all_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    print(f"データの保存先: {data_dir}")

    # 1. リスト取得
    print("[1/3] S&P500リスト取得...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        sp500_df = pd.read_html(io.StringIO(response.text))[0]
        tickers = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()
        
        # ★ここを修正: 全銘柄を対象にする
        target_tickers = tickers 
        print(f"本番モード: S&P500 全{len(target_tickers)}銘柄を処理します")

    except Exception as e:
        print(f"リスト取得エラー: {e}")
        return

    # 2. 株価取得
    print("[2/3] 株価データ取得 (過去10年分)...")
    try:
        data = yf.download(target_tickers, start="2015-01-01", end="2026-01-01", auto_adjust=True, progress=True)
        
        if isinstance(data.columns, pd.MultiIndex) and 'Close' in data.columns.levels[0]:
            data = data['Close']
        elif 'Close' in data.columns:
            data = data['Close']
        
        data.to_csv(os.path.join(data_dir, 'sp500_stock_prices.csv'))
    except Exception as e:
        print(f"株価取得エラー: {e}")

    # 3. 財務データ取得
    print("[3/3] 財務・配当データ取得...")
    fundamentals = []
    
    current_year = pd.Timestamp.now().year 
    
    for ticker in tqdm(target_tickers, mininterval=0.5):
        try:
            t = yf.Ticker(ticker)
            try:
                info = t.info
            except:
                info = {}
            
            # --- 10年増配率の計算 ---
            div_history = t.dividends
            div_growth_10y = 0.0
            
            if len(div_history) > 0:
                yearly_div = div_history.resample('YE').sum()
                
                if not yearly_div.empty and yearly_div.index[-1].year == current_year:
                    yearly_div = yearly_div.iloc[:-1]

                if len(yearly_div) >= 11:
                    current_div = yearly_div.iloc[-1] 
                    past_div = yearly_div.iloc[-11]
                    if past_div > 0:
                        div_growth_10y = (current_div / past_div) - 1.0
            # ---------------------------

            fundamentals.append({
                'Symbol': ticker,
                'Name': info.get('shortName'),
                'Sector': info.get('sector'),
                'Industry': info.get('industry'),
                'MarketCap': info.get('marketCap'),
                'DividendYield': info.get('dividendYield'),
                'PayoutRatio': info.get('payoutRatio'),
                'Beta': info.get('beta'),
                'OperatingMargins': info.get('operatingMargins'),
                'ReturnOnEquity': info.get('returnOnEquity'),
                'DividendGrowth10Y': div_growth_10y
            })
            time.sleep(0.05)
        except Exception:
            continue

    if fundamentals:
        fund_df = pd.DataFrame(fundamentals)
        fund_df.to_csv(os.path.join(data_dir, 'sp500_fundamentals.csv'), index=False)
        print("✅ 全データの取得完了")
    else:
        print("❌ データが取得できませんでした")

if __name__ == "__main__":
    fetch_all_data()