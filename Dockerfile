# ベースイメージ: 軽量なPython 3.10
FROM python:3.10-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
# ★修正: software-properties-common を削除し、代わりに git を追加しました
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ライブラリの依存関係ファイルをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコード一式をコンテナ内にコピー
COPY . .

# Streamlitが使うポートを開放
EXPOSE 8501

# コンテナ起動時に実行するコマンド
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]