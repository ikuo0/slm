
# SLM 第二世代
32トークンまで考慮できる言語モデルを作成
単語ベクトルは重み付きONE-HOTの総和とすることで順番を考慮できるようにする
加算による順番情報の消失は許容する
重みを設定するインデックスは単語のハッシュ値とし、単語インデックスを使わない
LM-A, LM-B の２つを連結することで省メモリに言語モデル構築を可能にする
LM-A 自己参照ＮＮで8～32次元の潜在ベクトルを作成する
LM-B は LM-A の出力を入力とし、次の単語を予測するモデルとする

# 重み付き ONE-HOT とは
予測単語からの距離１であれば１
２であれば１／２
３であれば１／３
…というように距離が遠いほど小さくなるように重みを設定する

# サンプルと設計
- chABSA サンプルを使用して諸々の試験を行い、言語モデルとして機能するかを確認する
- 圧縮単語ベクトルは２５６次元とする
  - 単語ハッシュ % 256 がベクトルのインデックスとなる
- LM-A について
  - 学習形態
    - Xデータを単語ベクトルとし
    - 中間ベクトル（潜在ベクトル）を8~32とし
    - YデータをXデータとする
    - 一度圧縮したベクトルを元の単語列に戻すことで文脈的な潜在ベクトルが生成されることを期待する
  - 潜在ベクトルの次元数
    - 8, 16, 32 を試す
- LM-B について
  - 学習形態
    - Xデータを LM-A の出力とする、次元数は LM-A の出力に従うので固定しなくて良い
    - Yデータを次の単語の単語ベクトルとする


# chABSA ダウンロードコマンド
※ 注意点
- `unzip` はパイプ（標準入力）から展開できないため、一度ファイルに保存してから展開する
- 本家リポジトリ（chakki-works/chABSA-dataset）の ZIP にはデータ本体が含まれず、公式配布元の S3 も閉鎖済みのため、データ入りミラーの sinjorjob/chABSA-dataset を使用する

```bash
# 保存先ディレクトリを作成
mkdir -p tmpl_gen1/samples/chabsa_data

# ZIP を一時ファイルに保存し、data/chABSA 配下の JSON のみ抽出してフラットに展開
curl -L -o /tmp/chABSA-dataset.zip https://github.com/sinjorjob/chABSA-dataset/archive/refs/heads/master.zip
unzip -j -o /tmp/chABSA-dataset.zip "chABSA-dataset-master/data/chABSA/*.json" -d ./tmpl_gen1/samples/chabsa_data/
rm /tmp/chABSA-dataset.zip

find tmpl_gen1/samples/chabsa_data -maxdepth 1 -type f -name '*.json' | wc -l
```
