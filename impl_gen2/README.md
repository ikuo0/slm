# SLM 第二世代 (impl_gen2)

32トークンまで考慮できる言語モデルを作成する（Xデータを任意の指定次元数 `vector_dim`（デフォルト 256 次元）にハッシュ圧縮する世代）。
単語ベクトルは「重み付き ONE-HOT の総和」とすることで順番（距離）を考慮できるようにする。
加算およびハッシュ衝突による情報の消失は「文脈32単語のインデックス組み合わせまで全く同じになることは稀である」という前提のもとで許容する。
重みを設定するインデックスには語彙インデックスをそのまま使わず、`単語ハッシュ値 % vector_dim`（デフォルト `vector_dim=256`、64〜256 等を `--vector-dim` オプションで指定可能）を使用する。

## 重み付き ONE-HOT（ハッシュ圧縮）とは
文脈（32単語）の直後にくる33単語目を予測対象の正解単語とし、正解単語からの距離 $d$ の逆数（$1/d$）を `単語ハッシュ値 % vector_dim` のインデックスに加算する。
- 予測単語からの距離 1 であれば $1$
- 距離 2 であれば $1/2$
- 距離 3 であれば $1/3$
- …
- 距離 32 であれば $1/32$

というように、距離が遠いほど小さくなるように重みを設定し、32単語分を足し合わせた `vector_dim` 次元ベクトルを1文脈の特徴量とする。
なお、プロセス間でハッシュ値が変わらないよう、単語ハッシュ値には MD5 ダイジェスト先頭 8 バイトの符号なし64bit整数（決定論的ハッシュ）を用いる。

## サンプルと設計
- chABSA サンプルを使用して諸々の試験を行い、言語モデルとして機能するかを確認する
- 圧縮単語ベクトルの次元数は **`--vector-dim`（別名 `--compress-dim`、デフォルト 256）** で指定可能（例: 64〜256）
  - `単語ハッシュ値 % vector_dim` がベクトルのインデックスとなる
- LM について
  - 学習形態（全データでのフルトレーニング）
    - Xデータ：圧縮単語ベクトル（32トークンの重み付き ONE-HOT 総和、`vector_dim` 次元の CSR 疎行列 `.npz`）
    - Yデータ：次の単語（33単語目の正解トークンID、値域 `0 .. vocab_size - 1` の1次元配列 `.npy`）
    - NN構造：入力層 `vector_dim` 次元 $\rightarrow$ 隠れ層 $\rightarrow$ 出力層 `vocab_size` 次元（Softmax）

## パイプライン実行例 (`phase3_featurize` 〜 `phase5_inference`)

```bash
# Phase 3: 特徴化 (--vector-dim で圧縮次元数を指定、デフォルト 256)
PYTHONPATH=impl_gen2/src/pipelines/training/phase3_featurize \
python -m phase3_featurize \
  --input-context-file=/tmp/job1234/phase2_tokenize/context.json \
  --out-x-file=/tmp/job1234_gen2/phase3_featurize/x.npz \
  --out-y-file=/tmp/job1234_gen2/phase3_featurize/y.npy \
  --out-meta-file=/tmp/job1234_gen2/phase3_featurize/metadata.json \
  --out-context-file=/tmp/job1234_gen2/phase3_featurize/context.json \
  --vector-dim=256 \
  --context-length=32

# Phase 4: モデル学習 (入力 vector_dim 次元 -> 出力 vocab_size 次元)
PYTHONPATH=impl_gen2/src/pipelines/training/phase4_training \
python -m phase4_training \
  --input-context-file=/tmp/job1234_gen2/phase3_featurize/context.json \
  --work-directory=/tmp/job1234_gen2/phase4_training \
  --out-model-file=/tmp/job1234_gen2/phase4_training/model.keras \
  --out-meta-file=/tmp/job1234_gen2/phase4_training/metadata.json \
  --out-context-file=/tmp/job1234_gen2/phase4_training/context.json \
  --epochs=50 \
  --min-delta=0.001 \
  --patience=3

# Phase 5: 推論・文章生成 (context.json から vector_dim を自動取得して推論)
PYTHONPATH=impl_gen2/src/pipelines/training/phase5_inference \
python -m phase5_inference \
  --input-context-file=/tmp/job1234_gen2/phase4_training/context.json \
  --out-texts-file=/tmp/job1234_gen2/phase5_inference/generated_texts.json \
  --out-readable-file=/tmp/job1234_gen2/phase5_inference/generated_texts.txt \
  --out-meta-file=/tmp/job1234_gen2/phase5_inference/metadata.json \
  --out-context-file=/tmp/job1234_gen2/phase5_inference/context.json \
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
mkdir -p impl_gen2/samples/chabsa_data/origin

# ZIP を一時ファイルに保存し、data/chABSA 配下の JSON のみ抽出してフラットに展開
curl -L -o /tmp/chABSA-dataset.zip https://github.com/sinjorjob/chABSA-dataset/archive/refs/heads/master.zip
unzip -j -o /tmp/chABSA-dataset.zip "chABSA-dataset-master/data/chABSA/*.json" -d ./impl_gen2/samples/chabsa_data/origin/
rm /tmp/chABSA-dataset.zip

find impl_gen2/samples/chabsa_data/origin -maxdepth 1 -type f -name '*.json' | wc -l
```
