# SLM 第三世代 (impl_gen3)

設計および機械学習ロジック（語彙サイズ $V$ 次元の重み付き ONE-HOT 総和）は第一世代（`impl_gen1`）と同じ。
サンプルに **livedoor ニュースコーパス** を加えて同様の学習・文章生成を行う。

そのため `phase1_preprocess` で chABSA と livedoor コーパスのフォーマット差を吸収し、共通構造の 1 つの JSON ファイルに保存してから正規化・後続フェーズへ引き継ぐ。

## テキストデータ
- **chABSA**
  - `samples/chabsa_data/data`（`*.json` の `sentences[].sentence` を抽出）
- **livedoor コーパス**
  - `samples/livedoor/text` の下の各種サブフォルダ（`LICENSE.txt` / `README.txt` / `CHANGES.txt` を除く記事 `*.txt` の 3 行目＝タイトルおよび 4 行目以降＝本文を抽出し、空行を除外）

## テキストデータ保存構造 (`slim.json`)

```json
{
  "sentences": [
    {
      "original": "chabsa/e00008_ann.json",
      "lines": [
        "元文章の各行"
      ]
    },
    {
      "original": "livedoor/topic-news/topic-news-5903225.txt",
      "lines": [
        "記事タイトル",
        "記事本文の各行"
      ]
    }
  ]
}
```

## 正規化後の構造 (`normalized.json`)

`テキストデータ保存構造` に同じ（各エントリの `lines` 内の文字列を `text_normalizer` で正規化したもの）。

## パイプライン実行例 (`phase1_preprocess` 〜 `phase5_inference`)

どのディレクトリからでも一括実行できるバッチスクリプトを用意しています：

```bash
bash /workspaces/slm/impl_gen3/src/batches/run_all.sh
```

個別に実行する場合：

```bash
# Phase 1: 前処理（chABSA + livedoor の共通フォーマット抽出・正規化）
PYTHONPATH=impl_gen3/src/pipelines/training/phase1_preprocess \
python -m phase1_preprocess \
  --chabsa-directory=samples/chabsa_data/data \
  --livedoor-directory=samples/livedoor/text \
  --out-slim-file=/tmp/job_gen3_1234/phase1_preprocess/slim.json \
  --out-normalized-file=/tmp/job_gen3_1234/phase1_preprocess/normalized.json \
  --out-meta-file=/tmp/job_gen3_1234/phase1_preprocess/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase1_preprocess/context.json

# Phase 2: トークン化・語彙サイズ探索
PYTHONPATH=impl_gen3/src/pipelines/training/phase2_tokenize \
python -m phase2_tokenize \
  --input-context-file=/tmp/job_gen3_1234/phase1_preprocess/context.json \
  --out-directory=/tmp/job_gen3_1234/phase2_tokenize \
  --out-meta-file=/tmp/job_gen3_1234/phase2_tokenize/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase2_tokenize/context.json \
  --vocab-size-start=1500 \
  --vocab-size-end=8000 \
  --vocab-size-exponent=1.03 \
  --score-threshold=0.70 \
  --model-type=unigram \
  --character-coverage=0.9995 \
  --user-defined-symbols='<s>,</s>,<n>' \
  --output-format=piece

# Phase 3: 特徴化
PYTHONPATH=impl_gen3/src/pipelines/training/phase3_featurize \
python -m phase3_featurize \
  --input-context-file=/tmp/job_gen3_1234/phase2_tokenize/context.json \
  --out-x-file=/tmp/job_gen3_1234/phase3_featurize/x.npz \
  --out-y-file=/tmp/job_gen3_1234/phase3_featurize/y.npy \
  --out-meta-file=/tmp/job_gen3_1234/phase3_featurize/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase3_featurize/context.json \
  --context-length=32

# Phase 4: モデル学習
PYTHONPATH=impl_gen3/src/pipelines/training/phase4_training \
python -m phase4_training \
  --input-context-file=/tmp/job_gen3_1234/phase3_featurize/context.json \
  --work-directory=/tmp/job_gen3_1234/phase4_training \
  --out-model-file=/tmp/job_gen3_1234/phase4_training/model.keras \
  --out-meta-file=/tmp/job_gen3_1234/phase4_training/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase4_training/context.json \
  --epochs=50 \
  --min-delta=0.001 \
  --patience=3

# Phase 5: 推論・文章生成
PYTHONPATH=impl_gen3/src/pipelines/training/phase5_inference \
python -m phase5_inference \
  --input-context-file=/tmp/job_gen3_1234/phase4_training/context.json \
  --out-texts-file=/tmp/job_gen3_1234/phase5_inference/generated_texts.json \
  --out-readable-file=/tmp/job_gen3_1234/phase5_inference/generated_texts.txt \
  --out-meta-file=/tmp/job_gen3_1234/phase5_inference/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase5_inference/context.json \
  --num-documents=10 \
  --max-chars=1000 \
  --sampling-mode=random \
  --prob-power=2.0
```
