# 専門用語抽出ツール

Markdownファイルから専門用語を抽出し、WikipediaやGrokで解説を取得するFlaskアプリケーションです。

## 機能

- Markdownファイルのアップロード
- Grok APIを使用した専門用語の自動抽出
- Wikipedia APIからの用語解説取得
- Grokによる解説生成（Wikipediaにない場合）
- レスポンシブなWebインターフェース

## セットアップ

1. リポジトリをクローン:

```bash
git clone https://github.com/your-repo/md-grok-wiki.git
cd md-grok-wiki
```

2. 仮想環境を作成して有効化:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. 依存関係をインストール:

```bash
pip install -r requirements.txt
```

4. 環境変数を設定:

```bash
export GROK_API_KEY="your-api-key-here"  # Linux/Mac
set GROK_API_KEY="your-api-key-here"  # Windows
```

## 実行方法

```bash
python app.py
```

アプリケーションは `http://localhost:5000` で起動します。

## 環境変数

| 変数名 | 説明 | 必須 |
|--------|------|------|
| GROK_API_KEY | Grok APIのアクセスキー | はい |

## 使用技術

- Python 3.x
- Flask (Webフレームワーク)
- python-frontmatter (Markdown解析)
- wikipedia-api (Wikipedia APIクライアント)
- requests (HTTPリクエスト)

## ライセンス

MIT License
# omniwiki-md
