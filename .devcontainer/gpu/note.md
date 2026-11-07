Analyzed devcontainer.json:L1-9
Analyzed devcontainer.json:L1-9
Analyzed requirements.txt:L1-4
Created devcontainer.json

[.devcontainer/gpu/devcontainer.json](file:///workspaces/slm/.devcontainer/gpu/devcontainer.json) を作成しました。

### 設定のポイント（Windows 11 + Docker Desktop WSL2 バックエンド向け）

- **`"runArgs": ["--gpus", "all"]`**：Windows 11 ホスト側の NVIDIA ドライバ（WSL2 経由）をコンテナへパススルーします。
- **`"containerEnv"`**：
  - `NVIDIA_VISIBLE_DEVICES=all` / `NVIDIA_DRIVER_CAPABILITIES=compute,utility`：コンテナ内にホストの `libcuda.so` や `nvidia-smi` をマウントさせます。
  - `TF_FORCE_GPU_ALLOW_GROWTH=true`：TensorFlow が起動時に GPU メモリ（VRAM）を100%専有してしまうのを防ぎ、必要な分だけ確保するようにします。
- **ベースイメージ・マウント**：CPU版（[.devcontainer/cpu/devcontainer.json](file:///workspaces/slm/.devcontainer/cpu/devcontainer.json)）と同じ `mcr.microsoft.com/devcontainers/python:3.11-bookworm` および `slm-venvs` ボリューム（`/opt/venvs`）を維持しています。

---

### コンテナ再起動後の GPU 用パッケージ補足

現在の TensorFlow（2.16 以降）は、`pip install tensorflow` 単体だと CPU 版のみが入り、`tensorflow[and-cuda]` を指定すると仮想環境（`/opt/venvs/.venv`）内に CUDA / cuDNN ランタイム一式が自動でインストールされます。

GPU コンテナで開いた後、仮想環境で以下を実行すると GPU が認識されます：

```bash
source /opt/venvs/.venv/bin/activate
pip install "tensorflow[and-cuda]"

# GPU 認識確認
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```