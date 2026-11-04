# 機械学習開発環境

Windows ホスト上の Docker Desktop（WSL 2 バックエンド、Linux コンテナー）と
VS Code の Dev Containers 拡張機能を使用します。

## 開始手順

1. Docker Desktop を起動します。
2. VS Code でこのフォルダーを開き、`Dev Containers: Reopen in Container` を実行します。
3. コンテナー内のターミナル（bash）で、仮想環境を作成して依存モジュールをインストールします。

```bash
python --version  # Python 3.11.x
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

新しいターミナルでは `source .venv/bin/activate` を実行してください。
VS Code で Python 拡張機能を使用する場合は、`Python: Select Interpreter` から
`.venv/bin/python` を選択してください。

コンテナーは Python 3.11 / Debian Bookworm の開発環境のみを用意します。
venv の作成と `requirements.txt` のインストールは自動実行しません。
依存モジュールのバージョンは固定していません。
GPU / CUDA の設定は含みません。
