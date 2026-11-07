"""Phase 3 特徴化パイプライン（重み付き One-Hot 特徴量 X と正解トークン ID Y の作成）。

Phase 2 が出力した context.json を元に、
- Phase 1 の正規化済み共通フォーマット JSON ファイル (normalized_file)
- Phase 2 で選定された最良語彙ファイル (vocab_file) およびモデルファイル (model_file)
を取得し、重み付き One-Hot 特徴量 X (.npz) と正解ラベル Y (.npy) を生成し、
metadata.json と次フェーズ用の context.json を出力する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# libs 配下のモジュールを単体実行時でも import できるようにパスを補完する
_LIBS_DIR = Path(__file__).resolve().parents[4] / "libs"
for _lib_name in ("weighted_onehot",):
    _lib_path = _LIBS_DIR / _lib_name
    if _lib_path.is_dir() and str(_lib_path) not in sys.path:
        sys.path.insert(0, str(_lib_path))

from weighted_onehot import (  # noqa: E402
    DEFAULT_BOS_TOKEN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EOS_TOKEN,
    DEFAULT_NEWLINE_TOKEN,
    build_and_save_xy,
)

__all__ = [
    "DEFAULT_BOS_TOKEN",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_EOS_TOKEN",
    "DEFAULT_NEWLINE_TOKEN",
    "load_context",
    "save_context",
    "resolve_phase1_normalized_path",
    "resolve_phase1_normalized_directory",
    "resolve_phase2_vocab_and_model",
    "run_phase3_featurize",
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


def resolve_phase1_normalized_path(context: dict) -> Path:
    """context.json から Phase 1 の正規化済み JSON ファイル (normalized_file または normalized_directory) を取得する。"""
    phase1 = context.get("phase1_preprocess", {})
    if isinstance(phase1, dict):
        if phase1.get("normalized_file"):
            return Path(phase1["normalized_file"])
        if phase1.get("normalized_directory"):
            return Path(phase1["normalized_directory"])
        meta_path_str = phase1.get("meta_file") or phase1.get("metadata")
        if meta_path_str:
            meta_path = Path(meta_path_str)
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            outputs = meta.get("outputs", {})
            norm_path = outputs.get("normalized_file") or outputs.get("normalized_directory")
            if norm_path:
                return Path(norm_path)

    phase2 = context.get("phase2_tokenize", {})
    if isinstance(phase2, dict):
        if phase2.get("normalized_file"):
            return Path(phase2["normalized_file"])
        if phase2.get("input_directory"):
            return Path(phase2["input_directory"])

    raise ValueError(
        "context.json 内に phase1_preprocess の normalized_file (または normalized_directory) パスが見つかりません"
    )


# 後方互換エイリアス
resolve_phase1_normalized_directory = resolve_phase1_normalized_path


def resolve_phase2_vocab_and_model(context: dict) -> tuple[Path, Path | None]:
    """context.json から Phase 2 で採用された vocab_file と model_file を取得する。"""
    phase2 = context.get("phase2_tokenize", {})
    if isinstance(phase2, dict):
        vocab_str = phase2.get("vocab_file")
        model_str = phase2.get("model_file")
        if vocab_str:
            return Path(vocab_str), (Path(model_str) if model_str else None)
        meta_path_str = phase2.get("meta_file") or phase2.get("metadata")
        if meta_path_str:
            meta_path = Path(meta_path_str)
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            outputs = meta.get("outputs", {})
            if outputs.get("vocab_file"):
                m_str = outputs.get("model_file")
                return Path(outputs["vocab_file"]), (Path(m_str) if m_str else None)

    raise ValueError(
        "context.json 内に phase2_tokenize の vocab_file パスが見つかりません"
    )


def run_phase3_featurize(
    input_context_file: str | Path,
    out_x_file: str | Path,
    out_y_file: str | Path,
    out_meta_file: str | Path,
    out_context_file: str | Path,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    newline_token: str = DEFAULT_NEWLINE_TOKEN,
    normalized_file: str | Path | None = None,
    normalized_directory: str | Path | None = None,
    vocab_file: str | Path | None = None,
    model_file: str | Path | None = None,
) -> dict:
    """Phase 2 の context.json を元に重み付き One-Hot 特徴量 X (.npz) と Y (.npy) を作成し、metadata.json と context.json を出力する。"""
    input_context_file = Path(input_context_file)
    context = load_context(input_context_file)

    override_norm = normalized_file if normalized_file is not None else normalized_directory
    resolved_norm_path = (
        Path(override_norm)
        if override_norm is not None
        else resolve_phase1_normalized_path(context)
    )

    if vocab_file is not None:
        resolved_vocab_file = Path(vocab_file)
        resolved_model_file = Path(model_file) if model_file is not None else None
    else:
        resolved_vocab_file, ctx_model_file = resolve_phase2_vocab_and_model(context)
        resolved_model_file = Path(model_file) if model_file is not None else ctx_model_file

    out_x_file = Path(out_x_file)
    out_y_file = Path(out_y_file)
    out_meta_file = Path(out_meta_file)
    out_context_file = Path(out_context_file)

    metadata = build_and_save_xy(
        resolved_norm_path,
        resolved_vocab_file,
        out_x_file,
        out_y_file,
        out_meta_file=None,
        model_file=resolved_model_file,
        context_length=context_length,
        bos_token=bos_token,
        eos_token=eos_token,
        newline_token=newline_token,
    )

    metadata["input"]["input_context_file"] = str(input_context_file)
    metadata["outputs"]["meta_file"] = str(out_meta_file)
    metadata["outputs"]["context_file"] = str(out_context_file)

    out_meta_file.parent.mkdir(parents=True, exist_ok=True)
    with out_meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    phase3_context = {
        "input_context_file": str(input_context_file),
        "normalized_file": str(resolved_norm_path),
        "vocab_file": str(resolved_vocab_file),
        "model_file": str(resolved_model_file) if resolved_model_file else None,
        "x_file": str(out_x_file),
        "y_file": str(out_y_file),
        "meta_file": str(out_meta_file),
    }

    context["phase3_featurize"] = phase3_context
    save_context(context, out_context_file)

    return metadata
