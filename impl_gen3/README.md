# SLM 第三世代 (impl_gen3)

設計および機械学習ロジック（語彙サイズ $V$ 次元の重み付き ONE-HOT 総和）は第一世代（`impl_gen1`）と同じ。
サンプルに **livedoor ニュースコーパス** および **青空文庫（aozora）** を加えて同様の学習・文章生成を行う。

そのため `phase1_preprocess` で chABSA・livedoor・青空文庫のフォーマット差を吸収し、共通構造の 1 つの UTF-8 JSON ファイルに保存してから正規化・後続フェーズへ引き継ぐ。

## テキストデータ
- **chABSA**
  - `samples/chabsa_data/data`（`*.json` の `sentences[].sentence` を抽出）
- **livedoor コーパス**
  - `samples/livedoor/text` の下の各種サブフォルダ（`LICENSE.txt` / `README.txt` / `CHANGES.txt` を除く記事 `*.txt` の 3 行目＝タイトルおよび 4 行目以降＝本文を抽出し、空行を除外）
- **青空文庫 (aozora)**
  - `samples/aozora` 配下の作品 `*.txt`（1ファイル＝1文書。UTF-8 / CP932 を自動判別して UTF-8 化し、冒頭メタデータ・末尾書誌情報・ルビ `《...》`・入力者注記 `［＃...］` を除去。外字 `※［＃...］` は JIS 面区点番号または U+ 表記から Unicode 文字へ変換し、変換不能な記号が残った作品は除外）

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
    },
    {
      "original": "aozora/cards/000005/files/5_ruby_21311/5_ruby_21311.txt",
      "lines": [
        "青空文庫の本文各行"
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
# Phase 1: 前処理（chABSA + livedoor + aozora の共通フォーマット抽出・正規化）
PYTHONPATH=impl_gen3/src/pipelines/training/phase1_preprocess \
python -m phase1_preprocess \
  --chabsa-directory=samples/chabsa_data/data \
  --livedoor-directory=samples/livedoor/text \
  --aozora-directory=samples/aozora \
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
  --out-directory=/tmp/job_gen3_1234/phase3_featurize \
  --shard-size=1000000 \
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

Phase 3 は100万サンプルごとに `x_0001.npz` / `y_0001.npy`、
`x_0002.npz` / `y_0002.npy` … を保存します。Phase 4 は context.json の
`x_files` / `y_files` の順序で1チャンクずつ読み込み、全チャンクで1エポックとします。
loss の収束判定は、全チャンクのサンプル数による加重平均を使ってエポック終了時だけ行います。

Phase 4 の再開は以下で実行できます。完了済みチャンクはスキップし、中断したチャンクの先頭から再開します。
再開には作業ディレクトリ内の `metadata.json` と `checkpoints/` を保持してください。
`model.keras` はエポック終了時に更新されます。

```bash
PYTHONPATH=impl_gen3/src/pipelines/training/phase4_training \
python -m phase4_training \
  --work-directory=/tmp/job_gen3_1234/phase4_training \
  --resume
```

### Phase 4 の疎入力計算

Phase 4 または `python -m nn_trainer` に `--input-mode sparse` を追加すると、
有効インデックスに対応する先頭層の重み行だけを取得し、入力値で重み付けして合計します。
全語彙次元の入力ベクトルへの展開を省略します。従来方式は `--input-mode dense`（既定値）です。
Python API では `run_phase4_training(..., input_mode="sparse")` または
`train_nn(..., input_mode="sparse")` で指定できます。

設定は metadata.json に保存され、`--resume` 時に復元されます。既存のメタデータは dense として扱います。
保存モデルは従来と同じ密ベクトル入力なので、Phase 5 の推論にもそのまま利用できます。
出力層は全語彙を計算するため、学習全体の高速化率は実測で比較してください。

### Phase 4 の CPU / GPU 切り替え

Phase 4 と `nn_trainer` の両方で `--device auto|cpu|gpu` を指定できます。
`auto`（既定）は利用可能なら GPU、`cpu` は CPU のみ、`gpu` は GPU 必須です。
GPU が利用できない環境で `gpu` を指定するとエラーになります。
起動時に選択結果を表示し、metadata.json に指定値と検出したデバイスを記録します。
GPU モードでも入力データの準備などは CPU で行います。

```bash
# 既存の学習を CPU のみで再開
PYTHONPATH=impl_gen3/src/pipelines/training/phase4_training \
python -m phase4_training \
  --work-directory=/tmp/job_gen3_1234/phase4_training \
  --resume --device cpu
```

`device` は再開時も今回の指定を使います（省略すると auto）。
Python API では `device="cpu"` などで指定します。
デバイスの切り替えは新しいプロセスで実行してください。

### 再開時のバッチサイズ変更

Phase 4 と `nn_trainer` は、再開時にも `--batch-size` を指定できます。
省略すると保存済みの値を維持します。新規学習で省略した場合は256です。

```bash
PYTHONPATH=impl_gen3/src/pipelines/training/phase4_training \
python -m phase4_training \
  --work-directory=/tmp/job_gen3_1234/phase4_training \
  --resume --device gpu --batch-size 2048
```

重みと optimizer 状態を引き継ぎ、次に学習するチャンクから新しいサイズを使います。
以後のチェックポイントに新しい値を保存します。Python API でも `batch_size=2048` を指定できます。
バッチサイズを変えると重みの更新回数や学習の進み方が変わるため、loss の推移も確認してください。
