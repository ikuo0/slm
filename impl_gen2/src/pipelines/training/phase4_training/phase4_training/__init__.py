"""Phase 4 モデル学習パイプライン（TensorFlow NN によるフルトレーニング・中断再開対応, impl_gen2）。

- 新規学習時 (resume=False):
  Phase 3 が出力した context.json からハッシュ圧縮 X (.npz, shape=(N, vector_dim)) と
  Y (.npy, shape=(N,)) のファイルパスおよび語彙サイズ (vocab_size) を取得し、
  nn_trainer で 1 エポックごとにモデルと metadata.json を保存しながら学習を行い、
  後続フェーズ（推論・検証等）へ引き継ぐ context.json を出力する。
- 中断再開時 (resume=True):
  他の引数を無視し、作業ディレクトリ内の metadata.json（および context.json）から
  設定を復元して続きのエポックから学習を再開する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# libs 配下のモジュールを単体実行時でも import できるようにパスを補完する
_LIBS_DIR = Path(__file__).resolve().parents[4] / "libs"
for _lib_name in ("nn_trainer",):
    _lib_path = _LIBS_DIR / _lib_name
    if _lib_path.is_dir() and str(_lib_path) not in sys.path:
        sys.path.insert(0, str(_lib_path))

from nn_trainer import (  # noqa: E402
    DEFAULT_ACTIVATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_UNITS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    train_nn,
)

__all__ = [
    "DEFAULT_ACTIVATION",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DROPOUT_RATE",
    "DEFAULT_EPOCHS",
    "DEFAULT_HIDDEN_UNITS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_PATIENCE",
    "load_context",
    "save_context",
    "resolve_phase3_xy_files",
    "resolve_vocab_size_from_context",
    "run_phase4_training",
]


def load_context(context_file: str | Path) -> dict:
    """context.json を読み込んで辞書として返す。"""
    path = Path(context_file)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, dict) else {}


def save_context(context: dict, out_context_file: str | Path) -> None:
    """ディレクトリ・ファイルのパス情報を格納した辞書を context.json に書き出す。"""
    path = Path(out_context_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)


def resolve_phase3_xy_files(context: dict) -> tuple[Path, Path]:
    """context.json から Phase 3 の x_file (.npz) と y_file (.npy) を取得する。"""
    phase3 = context.get("phase3_featurize", {})
    if isinstance(phase3, dict):
        x_str = phase3.get("x_file")
        y_str = phase3.get("y_file")
        if x_str and y_str:
            return Path(x_str), Path(y_str)
        meta_path_str = phase3.get("meta_file") or phase3.get("metadata")
        if meta_path_str:
            meta_path = Path(meta_path_str)
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            outputs = meta.get("outputs", {})
            if outputs.get("x_file") and outputs.get("y_file"):
                return Path(outputs["x_file"]), Path(outputs["y_file"])

    raise ValueError(
        "context.json 内に phase3_featurize の x_file / y_file パスが見つかりません"
    )


def resolve_vocab_size_from_context(context: dict) -> int | None:
    """context.json から語彙サイズ (vocab_size) を解決する。"""
    phase3 = context.get("phase3_featurize", {})
    if isinstance(phase3, dict):
        if phase3.get("vocab_size"):
            return int(phase3["vocab_size"])
        meta_path_str = phase3.get("meta_file") or phase3.get("metadata")
        if meta_path_str and Path(meta_path_str).is_file():
            with Path(meta_path_str).open(encoding="utf-8") as f:
                meta3 = json.load(f)
            vsize = meta3.get("stats", {}).get("vocab_size")
            if vsize:
                return int(vsize)

    phase2 = context.get("phase2_tokenize", {})
    if isinstance(phase2, dict):
        if phase2.get("selected_vocab_size"):
            return int(phase2["selected_vocab_size"])
        vocab_str = phase2.get("vocab_file")
        if vocab_str and Path(vocab_str).is_file():
            with Path(vocab_str).open(encoding="utf-8") as f:
                lines = [ln for ln in f if ln.rstrip("\r\n")]
            if lines:
                return len(lines)

    return None


def run_phase4_training(
    work_directory: str | Path,
    *,
    resume: bool = False,
    input_context_file: str | Path | None = None,
    out_model_file: str | Path | None = None,
    out_meta_file: str | Path | None = None,
    out_context_file: str | Path | None = None,
    vocab_size: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
    min_delta: float = DEFAULT_MIN_DELTA,
    patience: int = DEFAULT_PATIENCE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    hidden_units: tuple[int, ...] | list[int] = DEFAULT_HIDDEN_UNITS,
    activation: str = DEFAULT_ACTIVATION,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    seed: int = 42,
    x_file: str | Path | None = None,
    y_file: str | Path | None = None,
) -> dict:
    """Phase 3 の context.json を元に NN の学習を実行し、metadata.json と context.json を出力する。

    Args:
        work_directory: 作業ディレクトリ (model.keras, metadata.json, context.json のデフォルト保存先)
        resume: True の場合、他の引数を無視して作業ディレクトリの metadata.json から再開する
        input_context_file: Phase 3 が出力した context.json のファイルパス (resume=False 時は必須)
        out_model_file: モデルの出力ファイルパス (省略時は {work_directory}/model.keras)
        out_meta_file: メタデータの出力ファイルパス (省略時は {work_directory}/metadata.json)
        out_context_file: 次フェーズ用 context.json の出力ファイルパス (省略時は {work_directory}/context.json)
        vocab_size: 出力層の語彙サイズの上書き指定 (省略時は context.json から自動取得)
        epochs: 最大エポック数 (default: 50)
        min_delta: 収束判定の最小 Loss 改善幅 (default: 0.001)
        patience: 改善幅 < min_delta が何エポック連続したら収束終了とするか (default: 3)
        batch_size: ミニバッチサイズ (default: 256)
        learning_rate: 学習率 (default: 0.001)
        hidden_units: 隠れ層ユニット数 (default: (256, 256))
        activation: 隠れ層活性化関数 (default: "gelu")
        dropout_rate: ドロップアウト率 (default: 0.1)
        seed: 乱数シード (default: 42)
        x_file: 明示的に X (.npz) ファイルを上書き指定する場合のパス (任意)
        y_file: 明示的に Y (.npy) ファイルを上書き指定する場合のパス (任意)

    Returns:
        最新のメタデータ辞書
    """
    work_dir = Path(work_directory)
    default_context_path = work_dir / "context.json"

    if resume:
        # 再開時は他の引数を無視し、作業ディレクトリ内の metadata.json から復元して再開
        metadata = train_nn(
            work_dir,
            resume=True,
            out_meta_file=out_meta_file,
        )
        p_info = metadata.get("pipeline_info") or {}
        resolved_out_context = Path(p_info.get("context_file") or default_context_path)
        resolved_in_context = p_info.get("input_context_file")

        if resolved_out_context.is_file():
            context = load_context(resolved_out_context)
        elif resolved_in_context and Path(resolved_in_context).is_file():
            context = load_context(resolved_in_context)
        else:
            context = {}

        context["phase4_training"] = {
            "input_context_file": str(resolved_in_context) if resolved_in_context else None,
            "work_directory": str(work_dir),
            "x_file": metadata["input"]["x_file"],
            "y_file": metadata["input"]["y_file"],
            "vector_dim": metadata["stats"]["vector_dim"],
            "vocab_size": metadata["stats"]["vocab_size"],
            "model_file": metadata["outputs"]["model_file"],
            "meta_file": metadata["outputs"]["meta_file"],
        }
        save_context(context, resolved_out_context)
        if resolved_out_context.resolve() != default_context_path.resolve():
            save_context(context, default_context_path)
        return metadata

    # 新規学習時
    if input_context_file is None and (x_file is None or y_file is None):
        raise ValueError(
            "新規学習時 (resume=False) は input_context_file (または x_file と y_file) の指定が必須です"
        )

    context: dict = {}
    if input_context_file is not None:
        context = load_context(input_context_file)

    if x_file is not None and y_file is not None:
        resolved_x_file = Path(x_file)
        resolved_y_file = Path(y_file)
    else:
        resolved_x_file, resolved_y_file = resolve_phase3_xy_files(context)

    resolved_vocab_size = (
        int(vocab_size)
        if vocab_size is not None
        else resolve_vocab_size_from_context(context)
    )

    resolved_model_file = (
        Path(out_model_file) if out_model_file is not None else work_dir / "model.keras"
    )
    resolved_meta_file = (
        Path(out_meta_file) if out_meta_file is not None else work_dir / "metadata.json"
    )
    resolved_context_file = (
        Path(out_context_file) if out_context_file is not None else default_context_path
    )

    pipeline_info = {
        "input_context_file": str(input_context_file) if input_context_file is not None else None,
        "context_file": str(resolved_context_file),
    }

    phase4_context = {
        "input_context_file": str(input_context_file) if input_context_file is not None else None,
        "work_directory": str(work_dir),
        "x_file": str(resolved_x_file),
        "y_file": str(resolved_y_file),
        "vocab_size": resolved_vocab_size,
        "model_file": str(resolved_model_file),
        "meta_file": str(resolved_meta_file),
    }
    context["phase4_training"] = phase4_context

    # 途中で中断されても再開・参照できるよう学習開始前に context.json を書き出しておく
    save_context(context, resolved_context_file)
    if resolved_context_file.resolve() != default_context_path.resolve():
        save_context(context, default_context_path)

    metadata = train_nn(
        work_dir,
        resume=False,
        x_file=resolved_x_file,
        y_file=resolved_y_file,
        vocab_size=resolved_vocab_size,
        epochs=epochs,
        min_delta=min_delta,
        patience=patience,
        out_model_file=resolved_model_file,
        out_meta_file=resolved_meta_file,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        activation=activation,
        dropout_rate=dropout_rate,
        seed=seed,
        pipeline_info=pipeline_info,
    )

    phase4_context["vector_dim"] = metadata["stats"]["vector_dim"]
    phase4_context["vocab_size"] = metadata["stats"]["vocab_size"]
    context["phase4_training"] = phase4_context

    save_context(context, resolved_context_file)
    if resolved_context_file.resolve() != default_context_path.resolve():
        save_context(context, default_context_path)

    return metadata
