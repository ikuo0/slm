# SLM 第二世代 (impl_gen2)

32トークンまで考慮できる言語モデルを作成する（Xデータを 256 次元にハッシュ圧縮する世代）。
単語ベクトルは「重み付き ONE-HOT の総和」とすることで順番（距離）を考慮できるようにする。
加算およびハッシュ衝突による情報の消失は許容する。
重みを設定するインデックスには語彙インデックスを使わず、`単語ハッシュ値 % 256` を使用する。

## 重み付き ONE-HOT（256次元ハッシュ圧縮）とは
文脈（32単語）の直後にくる33単語目を予測対象の正解単語とし、正解単語からの距離 $d$ の逆数（$1/d$）を `単語ハッシュ値 % 256` のインデックスに加算する。
- 予測単語からの距離 1 であれば $1$
- 距離 2 であれば $1/2$
- 距離 3 であれば $1/3$
- …
- 距離 32 であれば $1/32$

というように、距離が遠いほど小さくなるように重みを設定し、32単語分を足し合わせた 256 次元ベクトルを1文脈の特徴量とする。

## サンプルと設計
- chABSA サンプルを使用して諸々の試験を行い、言語モデルとして機能するかを確認する
- 圧縮単語ベクトルは **256次元** とする
  - `単語ハッシュ値 % 256` がベクトルのインデックスとなる
- LM について
  - 学習形態
    - Xデータ：圧縮単語ベクトル（32トークンの重み付き ONE-HOT 総和、256次元）
    - Yデータ：次の単語（33単語目）

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
