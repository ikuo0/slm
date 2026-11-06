"""Phase 1 前処理パイプライン（データ簡素化・データ正規化）。

chABSA の origin JSON ディレクトリを入力として受け取り、
1. データ簡素化 (chabsa_slim)
2. データ正規化 (text_normalizer)
を順に実行して、各出力ディレクトリ・metadata.json・後続引き継ぎ用の context.json を出力する。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# libs 配下のモジュールを単体実行時でも import できるようにパスを補完する
_LIBS_DIR = Path(__file__).resolve().parents[4] / "libs"
for _lib_name in ("chabsa_slim", "text_normalizer"):
    _lib_path = _LIBS_DIR / _lib_name
    if _lib_path.is_dir() and str(_lib_path) not in sys.path:
        sys.path.insert(0, str(_lib_path))

from chabsa_slim import slim_directory  # noqa: E402
from text_normalizer import normalize_text  # noqa: E402

__all__ = [
    "load_context",
    "save_context",
    "normalize_document",
    "normalize_file",
    "normalize_directory",
    "run_phase1_preprocess",
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


def normalize_document(doc: dict) -> dict:
    """simplified 形式のドキュメント 1 件の sentence を正規化する。

    Args:
        doc: simplified JSON をパースした辞書 (metadata / sentences を持つ)

    Returns:
        sentence が正規化された simplified 形式の辞書
    """
    metadata = dict(doc.get("metadata", {}))
    sentences = [
        {
            "sentence_id": s["sentence_id"],
            "sentence": normalize_text(s["sentence"]),
        }
        for s in doc.get("sentences", [])
    ]
    return {"metadata": metadata, "sentences": sentences}


def normalize_file(in_path: str | Path, out_path: str | Path) -> dict:
    """simplified JSON ファイル 1 件を正規化して書き出す。変換結果を返す。"""
    in_path = Path(in_path)
    out_path = Path(out_path)
    with in_path.open(encoding="utf-8") as f:
        doc = json.load(f)
    normalized = normalize_document(doc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return normalized


def normalize_directory(in_dir: str | Path, out_dir: str | Path) -> dict:
    """ディレクトリ内の simplified *.json を列挙して一括正規化する。

    出力ファイル名は元のファイル名を踏襲する。

    Returns:
        書き出したファイルパス一覧・ファイル数・文数の辞書
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    written: list[Path] = []
    num_sentences = 0
    for in_path in sorted(in_dir.glob("*.json")):
        out_path = out_dir / in_path.name
        normalized = normalize_file(in_path, out_path)
        written.append(out_path)
        num_sentences += len(normalized.get("sentences", []))
    return {
        "written_files": written,
        "num_files": len(written),
        "num_sentences": num_sentences,
    }


def run_phase1_preprocess(
    input_directory: str | Path,
    out_slim_directory: str | Path,
    out_normalized_directory: str | Path,
    out_meta_file: str | Path,
    out_context_file: str | Path,
    *,
    input_context_file: str | Path | None = None,
    out_slim_meta_file: str | Path | None = None,
    out_normalized_meta_file: str | Path | None = None,
) -> dict:
    """データ簡素化とデータ正規化を順に実行し、metadata.json と context.json を出力する。

    Args:
        input_directory: chABSA origin JSON が格納された入力ディレクトリ
        out_slim_directory: データ簡素化後の JSON 出力ディレクトリ
        out_normalized_directory: データ正規化後の JSON 出力ディレクトリ
        out_meta_file: パイプライン全体のメタデータ出力ファイルパス (JSON)
        out_context_file: 後続フェーズへパスを引き継ぐ context.json の出力ファイルパス
        input_context_file: 前段の context.json がある場合に入力するファイルパス (任意)
        out_slim_meta_file: 簡素化ステップ単体のメタデータ出力ファイルパス (任意)
        out_normalized_meta_file: 正規化ステップ単体のメタデータ出力ファイルパス (任意)

    Returns:
        出力したメタデータの辞書
    """
    input_directory = Path(input_directory)
    out_slim_directory = Path(out_slim_directory)
    out_normalized_directory = Path(out_normalized_directory)
    out_meta_file = Path(out_meta_file)
    out_context_file = Path(out_context_file)

    total_started_at = datetime.now(timezone.utc)
    total_start = time.perf_counter()

    # 1. データ簡素化 (SR1100)
    slim_started_at = datetime.now(timezone.utc)
    slim_start = time.perf_counter()
    slim_written = slim_directory(input_directory, out_slim_directory)
    slim_elapsed = time.perf_counter() - slim_start
    slim_finished_at = datetime.now(timezone.utc)

    slim_meta = {
        "started_at": slim_started_at.isoformat(),
        "finished_at": slim_finished_at.isoformat(),
        "elapsed_seconds": round(slim_elapsed, 3),
        "input": {
            "input_directory": str(input_directory),
        },
        "outputs": {
            "slim_directory": str(out_slim_directory),
        },
        "stats": {
            "num_files": len(slim_written),
        },
    }
    if out_slim_meta_file is not None:
        slim_meta_path = Path(out_slim_meta_file)
        slim_meta["outputs"]["meta_file"] = str(slim_meta_path)
        slim_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with slim_meta_path.open("w", encoding="utf-8") as f:
            json.dump(slim_meta, f, ensure_ascii=False, indent=2)

    # 2. データ正規化 (SR1200)
    norm_started_at = datetime.now(timezone.utc)
    norm_start = time.perf_counter()
    norm_result = normalize_directory(out_slim_directory, out_normalized_directory)
    norm_elapsed = time.perf_counter() - norm_start
    norm_finished_at = datetime.now(timezone.utc)

    norm_meta = {
        "started_at": norm_started_at.isoformat(),
        "finished_at": norm_finished_at.isoformat(),
        "elapsed_seconds": round(norm_elapsed, 3),
        "input": {
            "slim_directory": str(out_slim_directory),
        },
        "outputs": {
            "normalized_directory": str(out_normalized_directory),
        },
        "stats": {
            "num_files": norm_result["num_files"],
            "num_sentences": norm_result["num_sentences"],
        },
    }
    if out_normalized_meta_file is not None:
        norm_meta_path = Path(out_normalized_meta_file)
        norm_meta["outputs"]["meta_file"] = str(norm_meta_path)
        norm_meta_path.parent.mkdir(parents=True, exist_ok=True)
        with norm_meta_path.open("w", encoding="utf-8") as f:
            json.dump(norm_meta, f, ensure_ascii=False, indent=2)

    total_elapsed = time.perf_counter() - total_start
    total_finished_at = datetime.now(timezone.utc)

    outputs_dict = {
        "slim_directory": str(out_slim_directory),
        "normalized_directory": str(out_normalized_directory),
        "meta_file": str(out_meta_file),
        "context_file": str(out_context_file),
    }
    if out_slim_meta_file is not None:
        outputs_dict["slim_meta_file"] = str(Path(out_slim_meta_file))
    if out_normalized_meta_file is not None:
        outputs_dict["normalized_meta_file"] = str(Path(out_normalized_meta_file))

    metadata = {
        "started_at": total_started_at.isoformat(),
        "finished_at": total_finished_at.isoformat(),
        "elapsed_seconds": round(total_elapsed, 3),
        "input": {
            "input_directory": str(input_directory),
        },
        "outputs": outputs_dict,
        "steps": {
            "slim": slim_meta,
            "normalize": norm_meta,
        },
        "stats": {
            "num_files": norm_result["num_files"],
            "num_sentences": norm_result["num_sentences"],
        },
    }

    out_meta_file.parent.mkdir(parents=True, exist_ok=True)
    with out_meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # context.json の構築・出力（ディレクトリ・ファイルのパスのみを格納し、文章等の実体は持たない）
    context: dict = {}
    if input_context_file is not None:
        context = load_context(input_context_file)

    phase1_context = {
        "input_directory": str(input_directory),
        "slim_directory": str(out_slim_directory),
        "normalized_directory": str(out_normalized_directory),
        "meta_file": str(out_meta_file),
    }
    if out_slim_meta_file is not None:
        phase1_context["slim_meta_file"] = str(Path(out_slim_meta_file))
    if out_normalized_meta_file is not None:
        phase1_context["normalized_meta_file"] = str(Path(out_normalized_meta_file))

    context["phase1_preprocess"] = phase1_context
    save_context(context, out_context_file)

    return metadata

