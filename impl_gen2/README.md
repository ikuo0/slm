# SLM 第二世代 (impl_gen2)

32トークンまで考慮できる言語モデルを作成する（Xデータを任意の指定次元数 `vector_dim`（デフォルト 256 次元）に 3 つのハッシュインデックスで圧縮する世代）。
単語ベクトルは「重み付き ONE-HOT の総和」とすることで順番（距離）を考慮できるようにする。
加算およびハッシュ衝突による情報の消失は「1単語あたり3つの独立したハッシュインデックス（$256^3 = 16,777,216$ 通り）および文脈32単語のインデックス組み合わせまで全く同じになることは稀である」という前提のもとで許容する。
重みを設定するインデックスには語彙インデックスをそのまま使わず、3つの決定論的ハッシュ関数 `hash1(word) % vector_dim`、`hash2(word) % vector_dim`、`hash3(word) % vector_dim`（デフォルト `vector_dim=256`, `num_hashes=3`）を使用する。

## 重み付き ONE-HOT（3ハッシュ圧縮）とは
文脈（32単語）の直後にくる33単語目を予測対象の正解単語とし、正解単語からの距離 $d$ の逆数（$\text{weight} = 1/d$）を、各単語の3つのハッシュインデックスに加算する。

```python
h1idx = hash1(word) % 256
h2idx = hash2(word) % 256
h3idx = hash3(word) % 256
vec256[h1idx] += weight
vec256[h2idx] += weight
vec256[h3idx] += weight
```

- 予測単語からの距離 1 であれば $1$
- 距離 2 であれば $1/2$
- 距離 3 であれば $1/3$
- …
- 距離 32 であれば $1/32$

というように、距離が遠いほど小さくなるように重みを設定し、32単語分（各単語につき3インデックス）を足し合わせた `vector_dim` 次元ベクトルを1文脈の特徴量とする。
なお、プロセス間でハッシュ値が変わらないよう、各ハッシュ関数（`hash1`, `hash2`, `hash3`）にはソルト付き SHA-256 ダイジェスト先頭 8 バイトの符号なし64bit整数（決定論的ハッシュ）を用いる。

## サンプルと設計
- chABSA サンプルを使用して諸々の試験を行い、言語モデルとして機能するかを確認する
- 圧縮単語ベクトルの次元数は **`--vector-dim`（別名 `--compress-dim`、デフォルト 256）** で指定可能（例: 64〜256）
- 1単語あたりのハッシュインデックス数は **`--num-hashes`（デフォルト 3）** で指定可能
  - `hash1(word) % vector_dim`, `hash2(word) % vector_dim`, `hash3(word) % vector_dim` がベクトルの加算先インデックスとなる
- LM について
  - 学習形態（全データでのフルトレーニング）
    - Xデータ：圧縮単語ベクトル（32トークン × 3ハッシュインデックスの重み付き ONE-HOT 総和、`vector_dim` 次元の CSR 疎行列 `.npz`）
    - Yデータ：次の単語（33単語目の正解トークンID、値域 `0 .. vocab_size - 1` の1次元配列 `.npy`）
    - NN構造：入力層 `vector_dim` 次元 $\rightarrow$ 隠れ層 $\rightarrow$ 出力層 `vocab_size` 次元（Softmax）

## パイプライン実行例 (`phase1_preprocess` 〜 `phase5_inference`)

どのディレクトリからでも一括実行できるバッチスクリプトを用意しています：

```bash
bash /workspaces/slm/impl_gen2/src/batches/run_all.sh
```

個別に実行する場合：

```bash
# Phase 1: 前処理（データ簡素化・正規化）
PYTHONPATH=impl_gen2/src/pipelines/training/phase1_preprocess \
python -m phase1_preprocess \
  --input-directory=samples/chabsa_data/data \
  --out-slim-directory=/tmp/job_gen2_1234/phase1_preprocess/slim \
  --out-normalized-directory=/tmp/job_gen2_1234/phase1_preprocess/normalized \
  --out-meta-file=/tmp/job_gen2_1234/phase1_preprocess/metadata.json \
  --out-context-file=/tmp/job_gen2_1234/phase1_preprocess/context.json

# Phase 2: トークン化・語彙サイズ探索
PYTHONPATH=impl_gen2/src/pipelines/training/phase2_tokenize \
python -m phase2_tokenize \
  --input-context-file=/tmp/job_gen2_1234/phase1_preprocess/context.json \
  --out-directory=/tmp/job_gen2_1234/phase2_tokenize \
  --out-meta-file=/tmp/job_gen2_1234/phase2_tokenize/metadata.json \
  --out-context-file=/tmp/job_gen2_1234/phase2_tokenize/context.json \
  --vocab-size-start=1500 \
  --vocab-size-end=8000 \
  --vocab-size-exponent=1.03 \
  --model-type=unigram \
  --character-coverage=0.9995 \
  --user-defined-symbols='<s>,</s>,<n>' \
  --output-format=piece

# Phase 3: 特徴化 (--vector-dim で圧縮次元数、--num-hashes でハッシュ関数数を指定)
PYTHONPATH=impl_gen2/src/pipelines/training/phase3_featurize \
python -m phase3_featurize \
  --input-context-file=/tmp/job_gen2_1234/phase2_tokenize/context.json \
  --out-x-file=/tmp/job_gen2_1234/phase3_featurize/x.npz \
  --out-y-file=/tmp/job_gen2_1234/phase3_featurize/y.npy \
  --out-meta-file=/tmp/job_gen2_1234/phase3_featurize/metadata.json \
  --out-context-file=/tmp/job_gen2_1234/phase3_featurize/context.json \
  --vector-dim=256 \
  --num-hashes=3 \
  --context-length=32

# Phase 4: モデル学習 (入力 vector_dim 次元 -> 出力 vocab_size 次元)
PYTHONPATH=impl_gen2/src/pipelines/training/phase4_training \
python -m phase4_training \
  --input-context-file=/tmp/job_gen2_1234/phase3_featurize/context.json \
  --work-directory=/tmp/job_gen2_1234/phase4_training \
  --out-model-file=/tmp/job_gen2_1234/phase4_training/model.keras \
  --out-meta-file=/tmp/job_gen2_1234/phase4_training/metadata.json \
  --out-context-file=/tmp/job_gen2_1234/phase4_training/context.json \
  --epochs=50 \
  --min-delta=0.001 \
  --patience=3

# Phase 5: 推論・文章生成 (context.json から vector_dim と num_hashes を自動取得して推論)
PYTHONPATH=impl_gen2/src/pipelines/training/phase5_inference \
python -m phase5_inference \
  --input-context-file=/tmp/job_gen2_1234/phase4_training/context.json \
  --out-texts-file=/tmp/job_gen2_1234/phase5_inference/generated_texts.json \
  --out-readable-file=/tmp/job_gen2_1234/phase5_inference/generated_texts.txt \
  --out-meta-file=/tmp/job_gen2_1234/phase5_inference/metadata.json \
  --out-context-file=/tmp/job_gen2_1234/phase5_inference/context.json \
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
