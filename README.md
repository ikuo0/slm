# 機械学習開発環境（SLM 実験プロジェクト）

chABSA データセットを使用し、32トークンの文脈（重み付き ONE-HOT の総和）から次の単語を予測する小規模言語モデル（SLM）を段階的に実験するプロジェクトです。

## 実験世代の構成

- **[impl_gen1](impl_gen1/README.md)（第一世代）**：X, Y を素直に作成
  - 語彙サイズ次元の重み付き ONE-HOT 総和をそのまま X データ（疎行列 `.npz`）とし、33単語目のトークンIDを Y データ（`.npy`）として学習するベースライン世代。
- **[impl_gen2](impl_gen2/README.md)（第二世代）**：X データを 256 次元にハッシュ圧縮
  - 語彙インデックスの代わりに `単語ハッシュ値 % 256` をインデックスとして X データを 256 次元に圧縮して学習する世代。
- **[impl_gen3](impl_gen3/README.md)（第三世代）**：LM-A / LM-B の 2 段階モデル
  - 256 次元の入力から 8〜32 次元の潜在ベクトルを作る **LM-A**（自己符号化器）と、その潜在ベクトルから次の単語を当てる **LM-B** の 2 段階構成世代。

---

## 開発環境と開始手順

Windows ホスト上の Docker Desktop（WSL 2 バックエンド、Linux コンテナー）と
VS Code の Dev Containers 拡張機能を使用します。

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

## bashメモ
```bash
# /opt/venvs/.venv を使う
source /opt/venvs/.venv/bin/activate
```
