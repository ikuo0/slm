# サンプルデータ統計集計 (`sandbox/sample_stats`)

正規化前・SentencePiece トークン化前のオリジナルデータ（`chABSA`、`livedoor` ニュースコーパス、および `aozora` 青空文庫）から、文書数・文（行）数・文字数・平均/標準偏差/最小/中央値/パーセンタイル/最大・文字数分布・文字種（ひらがな・カタカナ・漢字・英数字・全角/半角・NFKC変換対象文字など）の統計を集計するツールです。

青空文庫（`aozora`）については、文字コード（`cp932` / `utf-8-sig`）の内訳、外字注記 `※［＃...］` の Unicode 変換数、ルビ `《...》` や編集者注記 `［＃...］` の除去数、および変換不能・記法残存により除外（スキップ）された文書数と理由内訳もあわせて集計します。

## ディレクトリ構成

- `sample_stats/`
  - `__init__.py`: 集計コアロジック（`analyze_chabsa_directory`, `analyze_livedoor_directory`, `analyze_aozora_directory`, `analyze_corpora`, `format_summary_markdown`）
  - `__main__.py`: CLI エントリポイント（`python -m sample_stats`）
- `chabsa_stats.py`: chABSA 単体の統計集計スクリプト
- `livedoor_stats.py`: livedoor コーパス単体の統計集計スクリプト
- `aozora_stats.py`: 青空文庫（aozora）単体の統計集計スクリプト
- `run_stats.py`: chABSA・livedoor・aozora 全コーパス＋合算の一括統計集計スクリプト

## 実行方法

```bash
# 全コーパス（chABSA + livedoor + aozora）＋合算を一括集計
/opt/venvs/.venv/bin/python /workspaces/slm/sandbox/sample_stats/run_stats.py

# chABSA 単体を集計
/opt/venvs/.venv/bin/python /workspaces/slm/sandbox/sample_stats/chabsa_stats.py

# livedoor 単体を集計
/opt/venvs/.venv/bin/python /workspaces/slm/sandbox/sample_stats/livedoor_stats.py

# 青空文庫（aozora）単体を集計（全件、または特定ディレクトリ・上限指定）
/opt/venvs/.venv/bin/python /workspaces/slm/sandbox/sample_stats/aozora_stats.py
/opt/venvs/.venv/bin/python /workspaces/slm/sandbox/sample_stats/aozora_stats.py \
  --aozora-directory=/workspaces/slm/samples/aozora/cards/000005

# JSON や Markdown ファイルへ保存する場合
PYTHONPATH=/workspaces/slm/sandbox/sample_stats \
/opt/venvs/.venv/bin/python -m sample_stats \
  --out-json=/tmp/sample_stats.json \
  --out-md=/tmp/sample_stats.md
```
