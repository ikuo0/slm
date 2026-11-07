
SR = System Requirement

- ルール
  - 各 SR 単位で features に実装
  - 各 SR 及び F999 は context.yaml でデータを引き継いで処理をする
    - SR1000 -> SR1100 に context.yaml を引き継ぐ
    - SR1000 -> SR1000.F001 に context.yaml を引き継ぐ
    - こういった形式でデータの引き継ぎを行う
  - 前2桁でカテゴリとする
    - 後半2桁は予備番号
    - F999 は SR のサブ機能として扱う
  - libs に処理を実装
    - import して関数を使える、コマンドとしても単体で実行できるように作成する
  - features
    - 各 SR の処理を実装する単位
    - 主機能は libs なので、SR は呼び出し～記録、リソースアクセスを担当する
  - pipeline
    - 各 SR の処理を順番に実行する仕組み

- SR1000
  - パラメータ設定
- SR1100
  - パイプライン実行
- SR1300
  - 結果確認



- pipeline
  - phase1
    - 前処理
  - phase2
    - トークン化
  - phase3
    - 特徴化
  - phase4
    - モデル学習