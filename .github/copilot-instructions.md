# Copilot Instructions

このリポジトリは SLM（小規模言語モデル）の実験プロジェクトです。
chABSA データセットを使い、前処理 → 特徴化 → モデル学習のパイプラインを構築します。

## プロジェクト構成

- 設計の全体像は `impl_gen1/src/overview.md` を参照（SR 番号体系）
  - SR1??? = 前処理、SR2??? = 特徴化、SR3??? = モデル学習・検証
  - 各 SR / フェーズは `context.json` でパス情報を引き継ぐ（文章等の実体は持たない）
- `impl_gen?/` が実装世代ディレクトリ（例: `impl_gen1/`）
- 主機能は `impl_gen?/src/libs/` に実装し、pipelines / features は呼び出し・記録・リソースアクセスを担当

## libs モジュールの建付け（必須ルール）

- レイアウト: `src/libs/<name>/<name>/__init__.py` + `__main__.py`
  - `__init__.py`: 関数群を提供（`import <name>` で利用可能にする）
  - `__main__.py`: CLI エントリポイント（`python -m <name>` で実行）
    - argparse 実装・`main(argv)` 関数もここに置き、`if __name__ == "__main__":` ガードを付ける
  - **`main.py` は作らない**（`__main__.py` に統合する）
- 既存例: `chabsa_slim` / `text_normalizer` / `sp_tokenizer` / `chabsa_corpus`

## CLI の引数ポリシー（必須ルール）

- オプションは `--kebab-case` 形式
- **出力は 1 ファイルずつ個別のパスで指定させる**（`--out-model-file` など）
  - 出力のディレクトリ指定は禁止。複数出力が必要なら全部個別の引数にする
- 処理モジュールはメタデータ（開始終了時刻 UTC ISO 形式、処理時間、使用パラメータ、統計）を
  JSON で出力できるようにする（例: `sp_tokenizer` の `--out-meta-file`）

## テキスト正規化の仕様（text_normalizer）

- 全角統一（NFKC → 半角 ASCII を全角化）後に各種統一処理を行う
- 括弧系（開き/閉じペアがある系）はすべて `（ ）` に統一
- 引用符系（同じ記号で囲む系）はすべて `”` に統一
- **開き/閉じ（open/close）の対応は追跡しない仕様**

## SentencePiece / コーパスの仕様

- コーパスは **1 文書（= chABSA の JSON 1 ファイル）全体を `<s>` と `</s>` で囲み、文区切りを `<n>\n` として 1 文 = 1 行** で出力する
  - 例: `<s>文1<n>\n文2<n>\n文3</s>\n`
- 文中の生改行も特殊記号 `<n>` に置換する
- SentencePiece 学習時は `bos_id=-1, eos_id=-1` で組み込みの制御記号予約を解除した上で、`--user-defined-symbols='<s>,</s>,<n>'` として 3 つすべてを単一トークン登録する
- デコード後の `<n>` → `\n` 復元はアプリ側で行う

## データの取り扱い

- chABSA データ入手元: `sinjorjob/chABSA-dataset`（本家 ZIP にはデータ本体がなく、公式 S3 は閉鎖済み）
- `tmp/` `samples/` `_data/` は gitignore 対象。サンプルデータ（JSON）はコミットしない
- simplified 形式: `{"metadata": {...}, "sentences": [{"sentence_id", "sentence"}, ...]}`

## その他

- 依存追加時は `requirements.txt` を更新する
- 会話・ドキュメント・コード内コメントは日本語
