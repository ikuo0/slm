# 機械学習開発環境

Windows ホスト上の Docker Desktop（WSL 2 バックエンド、Linux コンテナー）と
VS Code の Dev Containers 拡張機能を使用します。

## 開始手順

1. Docker Desktop を起動します。
2. VS Code でこのフォルダーを開き、`Dev Containers: Reopen in Container` を実行します。
3. コンテナー内のターミナル（bash）で、仮想環境を作成して依存モジュールをインストールします。

```bash
python --version  # Python 3.11.x
python -m venv /opt/venvs/slm
source /opt/venvs/slm/bin/activate
python -m pip install -r requirements.txt
```

`/opt/venvs` は Docker の named volume `slm-venvs` にマウントされるため、
Windows ホストのプロジェクトフォルダーを経由せず、Docker Desktop の Linux
仮想ディスクへ直接保存されます。コンテナーを再作成しても仮想環境は保持されます。

仮想環境の作成、有効化、VS Code の Python インタープリター選択は手動で行います。
`requirements.txt` のインストールは自動実行しません。
依存モジュールのバージョンは固定していません。
GPU / CUDA の設定は含みません。


# bashメモ
```bash
# /opt/venvs/.venv を使う
source /opt/venvs/.venv/bin/activate
```