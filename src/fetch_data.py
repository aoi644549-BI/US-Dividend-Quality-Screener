import pandas as pd
import yfinance as yf
import requests
import io
import time
import os 
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'data')
os.makedirs(data_dir, exist_ok=True)

print(f"データの保存先: {data_dir}")


print("\n[1/3] S&P500の銘柄リストを取得中...")
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)
tables = pd.read_html(io.StringIO(response.text))
sp500_df = tables[0]

# ティッカー整形
tickers = sp500_df['Symbol'].str.replace('.', '-', regex=False).tolist()
target_tickers = tickers

print(f"取得完了: 全{len(tickers)}銘柄中、{len(target_tickers)}銘柄を処理します。")


print("\n[2/3] 株価データを取得中...")
data = yf.download(target_tickers, start="2019-01-01", end="2026-01-01", auto_adjust=True)
if 'Close' in data.columns.levels[0]:
    data = data['Close']
elif 'Close' in data.columns:
    data = data['Close']

price_csv_path = os.path.join(data_dir, 'sp500_stock_prices.csv')
data.to_csv(price_csv_path)
print(f"株価保存完了: {price_csv_path}")


print("\n[3/3] 財務データを取得中...")
fundamentals = []
for ticker in tqdm(target_tickers):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        data_dict = {
            'Symbol': ticker,
            'Name': info.get('shortName'),
            'Sector': info.get('sector'),
            'Price': info.get('currentPrice'),
            'MarketCap': info.get('marketCap'),
            'DividendYield': info.get('dividendYield'),
            'PayoutRatio': info.get('payoutRatio'),
            'Beta': info.get('beta'),
            'ForwardPE': info.get('forwardPE'),
            'FreeCashflow': info.get('freeCashflow'),
            'OperatingMargins': info.get('operatingMargins')
        }
        fundamentals.append(data_dict)
        time.sleep(0.5)
    except Exception as e:
        pass #

fund_df = pd.DataFrame(fundamentals)

fund_csv_path = os.path.join(data_dir, 'sp500_fundamentals.csv')
fund_df.to_csv(fund_csv_path, index=False)
print(f"財務データ保存完了: {fund_csv_path}")