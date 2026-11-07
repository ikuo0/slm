# SLM 第一世代 (impl_gen1)

32トークンまで考慮できる言語モデルを作成する（X, Y を語彙インデックスのまま素直に作成する世代）。
単語ベクトルは「重み付き ONE-HOT の総和」とすることで順番（距離）を考慮できるようにする。
加算による一部の順番情報の消失は許容する。

## 重み付き ONE-HOT とは
文脈（32単語）の直後にくる33単語目を予測対象の正解単語とし、正解単語からの距離 $d$ の逆数（$1/d$）を各単語の語彙インデックスに加算する。
- 予測単語からの距離 1 であれば $1$
- 距離 2 であれば $1/2$
- 距離 3 であれば $1/3$
- …
- 距離 32 であれば $1/32$

というように、距離が遠いほど小さくなるように重みを設定し、32単語分を足し合わせたベクトル（1サンプル＝語彙サイズ $V$ 次元の疎ベクトル）を1文脈の特徴量とする。
文書の先頭は32個の `<s>` でパディングし、文区切りは `<n>`、文書末尾は1個の `</s>` とする（末尾パディングは行わない）。

## サンプルと設計
- chABSA サンプルを使用して諸々の試験を行い、言語モデルとして機能するかを確認する
- LM について
  - 学習形態（全データでのフルトレーニング）
    - Xデータ：単語ベクトル（32トークンの重み付き ONE-HOT 総和、語彙サイズ $V$ 次元の CSR 疎行列 `.npz`）
    - Yデータ：次の単語（33単語目の正解トークンID、1次元配列 `.npy`）

## パイプライン実行例 (`phase1_preprocess` 〜 `phase5_inference`)

どのディレクトリからでも一括実行できるバッチスクリプトを用意しています：

```bash
bash /workspaces/slm/impl_gen1/src/batches/run_all.sh
```

個別に実行する場合：

```bash
# Phase 1: 前処理（データ簡素化・正規化）
PYTHONPATH=impl_gen1/src/pipelines/training/phase1_preprocess \
python -m phase1_preprocess \
  --input-directory=samples/chabsa_data/data \
  --out-slim-directory=/tmp/job_gen1_1234/phase1_preprocess/slim \
  --out-normalized-directory=/tmp/job_gen1_1234/phase1_preprocess/normalized \
  --out-meta-file=/tmp/job_gen1_1234/phase1_preprocess/metadata.json \
  --out-context-file=/tmp/job_gen1_1234/phase1_preprocess/context.json

# Phase 2: トークン化・語彙サイズ探索
PYTHONPATH=impl_gen1/src/pipelines/training/phase2_tokenize \
python -m phase2_tokenize \
  --input-context-file=/tmp/job_gen1_1234/phase1_preprocess/context.json \
  --out-directory=/tmp/job_gen1_1234/phase2_tokenize \
  --out-meta-file=/tmp/job_gen1_1234/phase2_tokenize/metadata.json \
  --out-context-file=/tmp/job_gen1_1234/phase2_tokenize/context.json \
  --vocab-size-start=1500 \
  --vocab-size-end=8000 \
  --vocab-size-exponent=1.03 \
  --model-type=unigram \
  --character-coverage=0.9995 \
  --user-defined-symbols='<s>,</s>,<n>' \
  --output-format=piece

# Phase 3: 特徴化
PYTHONPATH=impl_gen1/src/pipelines/training/phase3_featurize \
python -m phase3_featurize \
  --input-context-file=/tmp/job_gen1_1234/phase2_tokenize/context.json \
  --out-x-file=/tmp/job_gen1_1234/phase3_featurize/x.npz \
  --out-y-file=/tmp/job_gen1_1234/phase3_featurize/y.npy \
  --out-meta-file=/tmp/job_gen1_1234/phase3_featurize/metadata.json \
  --out-context-file=/tmp/job_gen1_1234/phase3_featurize/context.json \
  --context-length=32

# Phase 4: モデル学習
PYTHONPATH=impl_gen1/src/pipelines/training/phase4_training \
python -m phase4_training \
  --input-context-file=/tmp/job_gen1_1234/phase3_featurize/context.json \
  --work-directory=/tmp/job_gen1_1234/phase4_training \
  --out-model-file=/tmp/job_gen1_1234/phase4_training/model.keras \
  --out-meta-file=/tmp/job_gen1_1234/phase4_training/metadata.json \
  --out-context-file=/tmp/job_gen1_1234/phase4_training/context.json \
  --epochs=50 \
  --min-delta=0.001 \
  --patience=3

# Phase 5: 推論・文章生成
PYTHONPATH=impl_gen1/src/pipelines/training/phase5_inference \
python -m phase5_inference \
  --input-context-file=/tmp/job_gen1_1234/phase4_training/context.json \
  --out-texts-file=/tmp/job_gen1_1234/phase5_inference/generated_texts.json \
  --out-readable-file=/tmp/job_gen1_1234/phase5_inference/generated_texts.txt \
  --out-meta-file=/tmp/job_gen1_1234/phase5_inference/metadata.json \
  --out-context-file=/tmp/job_gen1_1234/phase5_inference/context.json \
  --num-documents=10 \
  --max-chars=1000 \
  --sampling-mode=random \
  --prob-power=2.0
```

## chABSA ダウンロードコマンド
※ 注意点
- `unzip` はパイプ（標準入力）から展開できないため、一度ファイルに保存してから展開する
- 本家リポジトリ（chakki-works/chABSA-dataset）の ZIP にはデータ本体が含まれず、公式配布元の S3 も閉鎖済みのため、データ入りミラーの `sinjorjob/chABSA-dataset` を使用する

```bash
# 保存先ディレクトリを作成
mkdir -p samples/chabsa_data/data

# ZIP を一時ファイルに保存し、data/chABSA 配下の JSON のみ抽出してフラットに展開
curl -L -o /tmp/chABSA-dataset.zip https://github.com/sinjorjob/chABSA-dataset/archive/refs/heads/master.zip
unzip -j -o /tmp/chABSA-dataset.zip "chABSA-dataset-master/data/chABSA/*.json" -d ./samples/chabsa_data/data/
rm /tmp/chABSA-dataset.zip

find samples/chabsa_data/data -maxdepth 1 -type f -name '*.json' | wc -l
```
