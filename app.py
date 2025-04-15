import flask
from flask import Flask, render_template, request, jsonify
# from pydantic import BaseModel, Field # Pydantic は直接使用しないため削除
# from typing import List, Optional # Typing も直接使用しないため削除
import aiohttp
import asyncio
from asyncio import Semaphore
import traceback # エラーハンドリング用

app = Flask(__name__)

# Grok APIエンドポイントURL
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

@app.route('/')
def index():
    """メインページを表示"""
    return render_template('index.html')

# /settings ルートは削除 (設定はクライアントサイド localStorage で管理)

@app.route('/analyze', methods=['POST'])
async def analyze_markdown():
    """Markdownテキストを受け取り、専門用語抽出と説明取得を行う"""
    # --- 1. リクエストデータの取得と検証 ---
    if not request.is_json:
        app.logger.warning("Invalid request format: not JSON")
        return jsonify({'error': 'リクエストはJSON形式である必要があります'}), 400

    data = request.get_json()
    text_content = data.get('text')
    model = data.get('model', 'grok-1') # デフォルトモデルを指定 (llama2-70bなども利用可能か確認)
    api_key = request.headers.get('X-Grok-API-Key') # ヘッダーからAPIキーを取得

    if not text_content:
        app.logger.warning("No text content provided")
        return jsonify({'error': 'テキストコンテンツが提供されていません'}), 400
    if not api_key:
        app.logger.warning("API key missing in header")
        return jsonify({'error': 'APIキーがリクエストヘッダー(X-Grok-API-Key)にありません'}), 401 # Unauthorized

    try:
        app.logger.info(f"分析開始 - 文字数: {len(text_content)}, モデル: {model}")

        # --- 2. Grokで専門用語抽出 ---
        app.logger.info("専門用語抽出開始...")
        extraction_result = await extract_technical_terms(text_content, api_key, model)

        if extraction_result.get('error'):
             # 抽出自体が失敗した場合
             app.logger.error(f"専門用語抽出エラー: {extraction_result['error']}")
             # エラー詳細に応じてステータスコードを返す
             status_code = 502 # Bad Gateway (Grok APIの問題の可能性)
             if 'Unauthorized' in extraction_result['error']:
                 status_code = 401
             elif 'Rate Limit' in extraction_result['error']:
                 status_code = 429
             return jsonify({'error': f"専門用語の抽出に失敗しました: {extraction_result['error']}"}), status_code

        terms = extraction_result.get('terms', [])
        if not terms:
            app.logger.info("専門用語が見つかりませんでした。")
            return jsonify({'results': []}) # 用語がない場合は空の結果を返す

        app.logger.info(f"専門用語抽出完了 (用語数: {len(terms)})")
        app.logger.debug(f"抽出された用語: {terms}")

        # --- 3. 各用語の説明を並列取得 ---
        app.logger.info("専門用語説明取得開始...")
        sem = Semaphore(5)  # 同時実行数を調整 (APIのレート制限に注意)
        async with aiohttp.ClientSession() as session:
            tasks = [get_term_explanation(term, session, sem, api_key, model) for term in terms]
            # asyncio.gather は結果のリストを返す (各要素は get_term_explanation の返り値辞書)
            explanations_results = await asyncio.gather(*tasks)

        # --- 4. 結果の整形 ---
        results = []
        for term, explanation_data in zip(terms, explanations_results):
            # explanation_data は {'source': 'grok', 'content': '...', 'error': None} 等の辞書
            results.append({
                'term': term,
                'explanation': explanation_data # そのまま explanation として設定
            })

        app.logger.info("全ての説明取得完了")
        return jsonify({'results': results})

    except Exception as e:
        error_trace = traceback.format_exc()
        app.logger.error(f"予期せぬエラー発生:\n{error_trace}")
        # クライアントには汎用的なエラーメッセージを返す
        return jsonify({
            'error': '分析処理中に予期せぬサーバーエラーが発生しました。',
            'details': str(e) if app.debug else None # デバッグモード時のみ詳細を追加
        }), 500


async def extract_technical_terms(text, api_key, model):
    """Grok APIを使用して専門用語を抽出 (APIキーとモデルを引数で受け取る)"""
    if not api_key:
        return {'terms': [], 'error': 'API key not provided to extraction function'}

    # プロンプト: 専門用語のみをカンマ区切りで返すよう指示
    system_prompt = "与えられたテキストから主要な専門用語を抽出し、他の説明や前置きは一切含めず、用語のみをカンマ(,)で区切ったリストとして返してください。例: 用語A, 用語B, 用語C"
    # 長すぎるテキストの処理 (例: 先頭 N 文字に限定)
    max_input_length = 8000 # APIのトークン制限に合わせて調整
    truncated_text = text[:max_input_length]
    if len(text) > max_input_length:
         app.logger.warning(f"入力テキストが長いため、最初の{max_input_length}文字に切り詰めました (用語抽出用)")

    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # "X-API-Key": api_key # Grok API がこちらを要求する場合もあるかもしれない
            }
            payload = {
                "model": model, # 引数で受け取ったモデルを使用
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": truncated_text}
                ],
                "max_tokens": 1000, # 用語リストなので、多すぎない程度に
                "temperature": 0.1, # 低温で安定したリスト出力を期待
                "stop": ["\n", "."] # リスト以外の出力を抑制する試み (効果はモデル次第)
            }

            app.logger.debug(f"Grok API呼び出し (抽出): model={model}, text_length={len(truncated_text)}")
            # タイムアウト設定を追加 (例: 60秒)
            async with session.post(GROK_API_URL, headers=headers, json=payload, timeout=60) as response:
                # ステータスコードチェック
                if response.status == 401:
                    app.logger.error("Grok API Unauthorized (401). Invalid API Key?")
                    return {'terms': [], 'error': 'Grok API Unauthorized (Invalid API Key?)'}
                if response.status == 429:
                    app.logger.warning("Grok API Rate Limit Exceeded (429).")
                    return {'terms': [], 'error': 'Grok API Rate Limit Exceeded'}
                # 他の 4xx, 5xx エラーもチェック
                response.raise_for_status()

                data = await response.json()

                # 応答データの検証
                if not data.get('choices') or not data['choices'][0].get('message') or not data['choices'][0]['message'].get('content'):
                     app.logger.error("Grok API応答形式エラー (抽出): contentが見つかりません")
                     return {'terms': [], 'error': 'Grok API response format error (no content)'}

                content = data['choices'][0]['message']['content']
                app.logger.debug(f"Grok抽出結果(raw): {content}")

                # 結果のクリーニングとパース
                # 前後の空白、引用符、不要な接頭辞（例: "抽出結果: "）を除去
                cleaned_content = content.strip().strip('"\'').replace("抽出結果: ", "").replace("専門用語: ", "")
                # カンマで分割し、各要素をクリーニング
                terms_list = [clean_term(term) for term in cleaned_content.split(',') if clean_term(term)]
                # 重複除去 (元の順序を維持)
                unique_terms = sorted(list(set(terms_list)), key=terms_list.index)

                app.logger.debug(f"パース後の用語: {unique_terms}")
                return {'terms': unique_terms, 'error': None}

    # エラーハンドリング
    except aiohttp.ClientResponseError as e:
        error_msg = f"Grok API HTTP Error (Extract): Status={e.status}, Message={e.message}"
        app.logger.error(error_msg)
        return {'terms': [], 'error': error_msg}
    except aiohttp.ClientError as e:
        error_msg = f"Grok API Connection Error (Extract): {str(e)}"
        app.logger.error(error_msg)
        return {'terms': [], 'error': error_msg}
    except asyncio.TimeoutError:
        error_msg = "Grok API request timed out (Extract)"
        app.logger.error(error_msg)
        return {'terms': [], 'error': error_msg}
    except Exception as e:
        error_msg = f"Unexpected Error during term extraction: {str(e)}"
        app.logger.error(error_msg)
        app.logger.error(traceback.format_exc())
        return {'terms': [], 'error': error_msg}

def clean_term(term):
    """専門用語文字列をクリーニングするヘルパー関数"""
    if not isinstance(term, str):
        return ""
    # Markdown強調を除去
    term = term.replace('**', '').replace('__', '')
    # 前後の不要な記号や空白を除去
    term = term.strip('*[](){}.,;:!?\'"“” ') # 全角記号も追加
    return term.strip()

async def get_term_explanation(term, session, sem, api_key, model):
    """特定の専門用語の説明をGrok APIで取得する (セマフォ制御付き)"""
    cleaned_term = clean_term(term)
    if not cleaned_term:
        app.logger.warning(f"無効な用語のため説明取得をスキップ: '{term}' -> '{cleaned_term}'")
        return {'source': 'none', 'content': '用語が無効か空です。', 'error': 'Invalid term provided'}

    app.logger.debug(f"説明取得開始: {cleaned_term} (Model: {model})")
    # get_grok_explanation を呼び出す
    return await get_grok_explanation(session, cleaned_term, sem, api_key, model)

async def get_grok_explanation(session, term, sem, api_key, model):
    """実際にGrok APIを呼び出して用語の説明を取得する関数"""
    if not api_key:
        return {'source': 'error', 'content': 'APIキーが提供されていません。', 'error': 'API key not provided to explanation function'}

    try:
        async with sem: # セマフォで同時実行数を制御
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # "X-API-Key": api_key
            }
            # プロンプト: 日本語で、簡潔に、初心者にも分かりやすく説明するよう指示
            system_prompt = "あなたは知識豊富なアシスタントです。指定された専門用語について、日本の読者（必ずしも専門家ではない）に向けて、簡潔かつ平易な日本語で説明してください。必要であればMarkdown形式を使用しても構いません。"
            user_prompt = f"「{term}」とは何ですか？分かりやすく説明してください。"

            payload = {
                "model": model, # 引数で受け取ったモデルを使用
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 350, # 説明なので少し長めに許可
                "temperature": 0.5 # ある程度の創造性を許容
            }

            app.logger.debug(f"Grok API呼び出し (説明): term={term}, model={model}")
            # タイムアウト設定 (例: 45秒)
            async with session.post(GROK_API_URL, headers=headers, json=payload, timeout=45) as response:
                # ステータスコードチェック
                if response.status == 401:
                    app.logger.error(f"Grok API Unauthorized (401) for term '{term}'.")
                    return {'source': 'error', 'content': f'Grok API認証エラー ({term})。', 'error': 'Grok API Unauthorized'}
                if response.status == 429:
                    app.logger.warning(f"Grok API Rate Limit Exceeded (429) for term '{term}'.")
                    # リトライはしない（必要なら別途実装）
                    await asyncio.sleep(1) # 短い待機
                    return {'source': 'error', 'content': f'Grok APIのレート制限を超えました ({term})。', 'error': 'Grok API Rate Limit Exceeded'}
                response.raise_for_status() # 他の 4xx, 5xx エラー

                data = await response.json()

                # 応答データの検証
                if not data.get('choices') or not data['choices'][0].get('message') or not data['choices'][0]['message'].get('content'):
                     app.logger.error(f"Grok API応答形式エラー (説明 - {term})")
                     return {'source': 'error', 'content': f'Grok APIからの応答形式が不正です ({term})。', 'error': 'Grok API response format error (no content)'}

                explanation_content = data['choices'][0]['message']['content'].strip()
                app.logger.debug(f"説明取得成功: {term} -> {explanation_content[:60]}...")
                # is_markdown フラグは不要 (クライアント側で md.render するため)
                return {
                    'source': 'grok', # ソースを明記
                    'content': explanation_content,
                    'error': None # エラーなし
                }

    # エラーハンドリング
    except aiohttp.ClientResponseError as e:
        error_msg = f"Grok API HTTP Error (Explain '{term}'): Status={e.status}, Message={e.message}"
        app.logger.error(error_msg)
        return {'source': 'error', 'content': f'Grok APIから「{term}」の説明取得中にHTTPエラー ({e.status})。', 'error': error_msg}
    except aiohttp.ClientError as e:
        error_msg = f"Grok API Connection Error (Explain '{term}'): {str(e)}"
        app.logger.error(error_msg)
        return {'source': 'error', 'content': f'Grok APIへの接続エラー ({term})。', 'error': error_msg}
    except asyncio.TimeoutError:
        error_msg = f"Grok API request timed out (Explain '{term}')"
        app.logger.error(error_msg)
        return {'source': 'error', 'content': f'Grok APIへのリクエストがタイムアウトしました ({term})。', 'error': error_msg}
    except Exception as e:
        error_msg = f"Unexpected Error during explanation fetching for '{term}': {str(e)}"
        app.logger.error(error_msg)
        app.logger.error(traceback.format_exc())
        return {'source': 'error', 'content': f'「{term}」の説明取得中に予期せぬエラーが発生しました。', 'error': error_msg}

if __name__ == '__main__':
    # デバッグモードで実行、ローカルネットワークからもアクセス可能にする場合は host='0.0.0.0' を指定
    app.run(debug=True, host='127.0.0.1', port=5000)