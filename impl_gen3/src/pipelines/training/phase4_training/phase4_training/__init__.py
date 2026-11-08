"""Phase 4 モデル学習パイプライン（TensorFlow NN によるフルトレーニング・中断再開対応）。

- 新規学習時 (resume=False):
  Phase 3 が出力した context.json から X (.npz) と Y (.npy) のファイルパスを取得し、
  nn_trainer で チャンクごとにモデルと metadata.json を保存しながら学習を行い、
  後続フェーズ（推論・検証等）へ引き継ぐ context.json を出力する。
- 中断再開時 (resume=True):
  device、batch_size と出力メタデータ指定以外の引数を無視し、作業ディレクトリ内の metadata.json（および context.json）から
  設定を復元して未完了チャンクから学習を再開する。
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


def resolve_phase3_xy_files(context: dict) -> tuple[list[Path], list[Path]]:
    """Phase 3 の順序付きチャンク一覧を取得する（旧単一ファイル形式も対応）。"""
    phase3 = context.get("phase3_featurize", {})
    if isinstance(phase3, dict):
        sources = [phase3]
        meta_path = phase3.get("meta_file") or phase3.get("metadata")
        if not (phase3.get("x_files") or phase3.get("x_file")) and meta_path:
            with Path(meta_path).open(encoding="utf-8") as f:
                sources.append(json.load(f).get("outputs", {}))
        for source in sources:
            xs, ys = source.get("x_files"), source.get("y_files")
            if xs is not None or ys is not None:
                if not xs or not ys or len(xs) != len(ys):
                    raise ValueError("Phase 3 の X/Y チャンク一覧が空または件数不一致です")
                return [Path(x) for x in xs], [Path(y) for y in ys]
            if source.get("x_file") and source.get("y_file"):
                return [Path(source["x_file"])], [Path(source["y_file"])]
    raise ValueError("context.json 内に Phase 3 の X/Y パスが見つかりません")


def run_phase4_training(
    work_directory: str | Path,
    *,
    resume: bool = False,
    input_context_file: str | Path | None = None,
    out_model_file: str | Path | None = None,
    out_meta_file: str | Path | None = None,
    out_context_file: str | Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    min_delta: float = DEFAULT_MIN_DELTA,
    patience: int = DEFAULT_PATIENCE,
    batch_size: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    hidden_units: tuple[int, ...] | list[int] = DEFAULT_HIDDEN_UNITS,
    activation: str = DEFAULT_ACTIVATION,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    input_mode: str = "dense",
    device: str = "auto",
    seed: int = 42,
    x_file: str | Path | None = None,
    y_file: str | Path | None = None,
) -> dict:
    """Phase 3 の context.json を元に NN の学習を実行し、metadata.json と context.json を出力する。

    Args:
        work_directory: 作業ディレクトリ (model.keras, metadata.json, context.json のデフォルト保存先)
        resume: True の場合、device、batch_size と出力メタデータ指定以外の引数を無視して作業ディレクトリの metadata.json から再開する
        input_context_file: Phase 3 が出力した context.json のファイルパス (resume=False 時は必須)
        out_model_file: モデルの出力ファイルパス (省略時は {work_directory}/model.keras)
        out_meta_file: メタデータの出力ファイルパス (省略時は {work_directory}/metadata.json)
        out_context_file: 次フェーズ用 context.json の出力ファイルパス (省略時は {work_directory}/context.json)
        epochs: 最大エポック数 (default: 50)
        min_delta: 収束判定の最小 Loss 改善幅 (default: 0.001)
        patience: 改善幅 < min_delta が何エポック連続したら収束終了とするか (default: 3)
        batch_size: 明示指定時は再開時も変更。省略時は新規256、再開時は保存値
        learning_rate: 学習率 (default: 0.001)
        hidden_units: 隠れ層ユニット数 (default: (256, 256))
        activation: 隠れ層活性化関数 (default: "gelu")
        dropout_rate: ドロップアウト率 (default: 0.1)
        input_mode: 先頭層の計算方式 (dense または sparse)
        device: 実行デバイス (auto, cpu, gpu)。再開時も指定可能
        seed: 乱数シード (default: 42)
        x_file: 明示的に X (.npz) ファイルを上書き指定する場合のパス (任意)
        y_file: 明示的に Y (.npy) ファイルを上書き指定する場合のパス (任意)

    Returns:
        最新のメタデータ辞書
    """
    work_dir = Path(work_directory)
    default_context_path = work_dir / "context.json"

    if resume:
        # 再開時は device、batch_size と出力メタデータ指定以外の引数を無視し、作業ディレクトリ内の metadata.json から復元して再開
        metadata = train_nn(
            work_dir,
            resume=True,
            batch_size=batch_size,
            device=device,
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
            "x_files": metadata["input"]["x_files"],
            "y_files": metadata["input"]["y_files"],
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
        resolved_x_files = [Path(x_file)]
        resolved_y_files = [Path(y_file)]
    else:
        if x_file is not None or y_file is not None:
            raise ValueError("x_file と y_file は両方指定してください")
        resolved_x_files, resolved_y_files = resolve_phase3_xy_files(context)

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
        "x_file": str(resolved_x_files[0]),
        "x_files": [str(p) for p in resolved_x_files],
        "y_file": str(resolved_y_files[0]),
        "y_files": [str(p) for p in resolved_y_files],
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
        x_files=resolved_x_files,
        y_files=resolved_y_files,
        epochs=epochs,
        min_delta=min_delta,
        patience=patience,
        out_model_file=resolved_model_file,
        out_meta_file=resolved_meta_file,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_units=hidden_units,
        input_mode=input_mode,
        device=device,
        activation=activation,
        dropout_rate=dropout_rate,
        seed=seed,
        pipeline_info=pipeline_info,
    )

    save_context(context, resolved_context_file)
    if resolved_context_file.resolve() != default_context_path.resolve():
        save_context(context, default_context_path)

    return metadata
