# SLM 第三世代 (impl_gen3)

32トークンまで考慮できる言語モデルを作成する（潜在ベクトルを作る LM-A と、潜在ベクトルから次の単語を当てる LM-B の2段階構成世代）。
単語ベクトルは「重み付き ONE-HOT の総和」とすることで順番（距離）を考慮できるようにし、インデックスには `単語ハッシュ値 % 256` を用いる。
- **LM-A**：自己符号化器（自己参照NN）で 256 次元の圧縮単語ベクトルから 8〜32 次元の潜在ベクトルを作成する
- **LM-B**：LM-A が出力した潜在ベクトルを入力とし、次の単語を予測するモデルとする
- LM-A と LM-B の2段階に分けることで、省メモリかつ文脈を凝縮した言語モデル構築を目指す

## 重み付き ONE-HOT（256次元ハッシュ圧縮）とは
文脈（32単語）の直後にくる33単語目を予測対象の正解単語とし、正解単語からの距離 $d$ の逆数（$1/d$）を `単語ハッシュ値 % 256` のインデックスに加算する。
- 予測単語からの距離 1 であれば $1$
- 距離 2 であれば $1/2$
- 距離 3 であれば $1/3$
- …
- 距離 32 であれば $1/32$

## サンプルと設計
- chABSA サンプルを使用して諸々の試験を行い、言語モデルとして機能するかを確認する
- 圧縮単語ベクトルは **256次元** とする
  - `単語ハッシュ値 % 256` がベクトルのインデックスとなる
- **LM-A（潜在ベクトル生成モデル）について**
  - 学習形態
    - Xデータ：圧縮単語ベクトル（256次元）
    - 中間ベクトル（潜在ベクトル）：8〜32次元
    - Yデータ：Xデータと同じ（自己復元）
    - 一度低次元に圧縮したベクトルから元のベクトルを復元させることで、文脈情報を凝縮した潜在ベクトルが獲得されることを期待する
  - 潜在ベクトルの次元数：8, 16, 32 を試す
- **LM-B（次単語予測モデル）について**
  - 学習形態
    - Xデータ：LM-A の中間層（潜在ベクトル）の出力（次元数は LM-A の設定に従う）
    - Yデータ：次の単語（33単語目）

## chABSA ダウンロードコマンド
※ 注意点
- `unzip` はパイプ（標準入力）から展開できないため、一度ファイルに保存してから展開する
- 本家リポジトリ（chakki-works/chABSA-dataset）の ZIP にはデータ本体が含まれず、公式配布元の S3 も閉鎖済みのため、データ入りミラーの `sinjorjob/chABSA-dataset` を使用する

```bash
# 保存先ディレクトリを作成
mkdir -p impl_gen3/samples/chabsa_data/origin

# ZIP を一時ファイルに保存し、data/chABSA 配下の JSON のみ抽出してフラットに展開
curl -L -o /tmp/chABSA-dataset.zip https://github.com/sinjorjob/chABSA-dataset/archive/refs/heads/master.zip
unzip -j -o /tmp/chABSA-dataset.zip "chABSA-dataset-master/data/chABSA/*.json" -d ./impl_gen3/samples/chabsa_data/origin/
rm /tmp/chABSA-dataset.zip

find impl_gen3/samples/chabsa_data/origin -maxdepth 1 -type f -name '*.json' | wc -l
```
