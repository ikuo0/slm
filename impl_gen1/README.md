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

## chABSA ダウンロードコマンド
※ 注意点
- `unzip` はパイプ（標準入力）から展開できないため、一度ファイルに保存してから展開する
- 本家リポジトリ（chakki-works/chABSA-dataset）の ZIP にはデータ本体が含まれず、公式配布元の S3 も閉鎖済みのため、データ入りミラーの `sinjorjob/chABSA-dataset` を使用する

```bash
# 保存先ディレクトリを作成
mkdir -p impl_gen1/samples/chabsa_data/origin

# ZIP を一時ファイルに保存し、data/chABSA 配下の JSON のみ抽出してフラットに展開
curl -L -o /tmp/chABSA-dataset.zip https://github.com/sinjorjob/chABSA-dataset/archive/refs/heads/master.zip
unzip -j -o /tmp/chABSA-dataset.zip "chABSA-dataset-master/data/chABSA/*.json" -d ./impl_gen1/samples/chabsa_data/origin/
rm /tmp/chABSA-dataset.zip

find impl_gen1/samples/chabsa_data/origin -maxdepth 1 -type f -name '*.json' | wc -l
```
