
# SLM 第一世代
32トークンまで考慮できる言語モデルを作成
単語ベクトルは重み付きONE-HOTの総和とすることで順番を考慮できるようにする
加算による順番情報の消失は許容する
重みを設定するインデックスは単語のハッシュ値とし、単語インデックスを使わない


# 重み付き ONE-HOT とは
予測単語からの距離１であれば１
２であれば１／２
３であれば１／３
…というように距離が遠いほど小さくなるように重みを設定する

# サンプルと設計
- chABSA サンプルを使用して諸々の試験を行い、言語モデルとして機能するかを確認する
- 圧縮単語ベクトルは２５６次元とする
  - 単語ハッシュ % 256 がベクトルのインデックスとなる
- LM について
  - 学習形態
    - Xデータ
      - 単語ベクトル(32トークンの重み付きONE-HOT総和) 
    - Yデータ
      - 次の単語


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
