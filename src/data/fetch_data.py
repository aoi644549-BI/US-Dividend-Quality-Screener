%%writefile src/data/fetch_data.py
import pandas as pd
import yfinance as yf
import os
from tqdm import tqdm
import time
import requests 


# 保存先ディレクトリ
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_sp500_tickers():
    print("[1/3] S&P500リスト取得...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        # User-Agentヘッダーを追加して、ブラウザのふりをする
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() # エラーならここで止める
        
        # 取得したHTMLテキストをpandasに渡す
        dfs = pd.read_html(response.text)
        
        # 通常、最初のテーブルが構成銘柄リスト
        df = dfs[0]
        tickers = df['Symbol'].tolist()
        
        # Yahoo Finance用にシンボルを変換 (例: BF.B -> BF-B)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"  -> {len(tickers)} 銘柄を取得しました。")
        return tickers
        
    except Exception as e:
        print(f"リスト取得エラー: {e}")
        return []

def fetch_stock_prices(tickers):
    print("[2/3] 株価データ取得 (過去10年分)...")
    if not tickers:
        print("銘柄リストが空のためスキップします。")
        return
        
    # 全銘柄を一括ダウンロード（高速化）
    # auto_adjust=True で分割併合を考慮した株価を取得
    try:
        data = yf.download(tickers, period="10y", interval="1d", group_by='ticker', auto_adjust=True, threads=True)
        
        # CSVとして保存
        save_path = os.path.join(DATA_DIR, "sp500_stock_prices.csv")
        data.to_csv(save_path)
        print(f"  -> 株価データを保存しました: {save_path}")
    except Exception as e:
        print(f"株価データ取得エラー: {e}")

def fetch_fundamentals(tickers):
    print("[3/3] 財務・配当データ取得...")
    fundamentals = []
    
    # 財務データは1つずつ取る必要がある（API制限に注意）
    for ticker in tqdm(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 必要なデータだけ抽出
            data = {
                'Symbol': ticker,
                'Name': info.get('shortName', 'N/A'),
                'Sector': info.get('sector', 'Unknown'),
                'Industry': info.get('industry', 'Unknown'),
                'MarketCap': info.get('marketCap', 0),
                'DividendYield': info.get('dividendYield', 0), 
                'PayoutRatio': info.get('payoutRatio', 0),
                'OperatingMargins': info.get('operatingMargins', 0),
                'ReturnOnEquity': info.get('returnOnEquity', 0),
                'Beta': info.get('beta', 1.0),
                'DividendRate': info.get('dividendRate', 0),
            }
            
            # 10年増配率 (DividendGrowth10Y) 計算
            try:
                hist = stock.history(period="10y")
                dividends = hist['Dividends']
                if len(dividends) > 0:
                    div_yearly = dividends.resample('YE').sum() # 'Y' is deprecated, use 'YE'
                    if len(div_yearly) >= 10:
                        start_div = div_yearly.iloc[0]
                        end_div = div_yearly.iloc[-1]
                        if start_div > 0:
                            growth = (end_div / start_div) ** (1/10) - 1
                            data['DividendGrowth10Y'] = growth
                        else:
                            data['DividendGrowth10Y'] = 0
                    else:
                         data['DividendGrowth10Y'] = 0
                else:
                    data['DividendGrowth10Y'] = 0
            except:
                data['DividendGrowth10Y'] = 0

            fundamentals.append(data)
            
        except Exception as e:
            continue
            
    # データフレーム化して保存
    if fundamentals:
        df = pd.DataFrame(fundamentals)
        save_path = os.path.join(DATA_DIR, "sp500_fundamentals.csv")
        df.to_csv(save_path, index=False)
        print(f"  -> 財務データを保存しました: {save_path}")
    else:
        print("  -> 財務データが取得できませんでした。")

if __name__ == "__main__":
    # 1. リスト取得
    tickers = fetch_sp500_tickers()
    
    # 本番モード: 全銘柄実行
    target_tickers = tickers 
    
    if target_tickers:
        print(f"本番モード: S&P500 全{len(target_tickers)}銘柄を処理します")
        
        # 2. 株価取得
        fetch_stock_prices(target_tickers)
        
        # 3. 財務データ取得
        fetch_fundamentals(target_tickers)
        
        print("✅ 全データの取得完了")
    else:
        print("❌ 銘柄リストの取得に失敗したため終了します。")