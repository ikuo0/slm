"""Phase 1 前処理パイプライン（共通フォーマット抽出・テキスト正規化）。

chABSA の origin JSON ディレクトリ、livedoor コーパスのテキストディレクトリ、および青空文庫ディレクトリを入力として受け取り、
1. 共通フォーマットへの抽出・統合 (chabsa_slim.build_slim_dataset)
   {
     "sentences": [
       {
         "original": "chabsa/e00008_ann.json",
         "lines": ["元文章の各行..."]
       },
       {
         "original": "livedoor/topic-news/topic-news-xxx.txt",
         "lines": ["記事タイトル", "記事本文の各行..."]
       },
       {
         "original": "aozora/cards/000005/files/5_ruby_21311/5_ruby_21311.txt",
         "lines": ["青空文庫の本文各行..."]
       }
     ]
   }
2. テキスト正規化 (text_normalizer.normalize_text)
   共通フォーマットの構造を維持したまま lines の各行を正規化する
を順に実行して、slim.json・normalized.json・metadata.json・後続引き継ぎ用の context.json を UTF-8 で出力する。
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

from chabsa_slim import build_slim_dataset  # noqa: E402
from text_normalizer import normalize_text  # noqa: E402

__all__ = [
    "load_context",
    "save_context",
    "normalize_dataset",
    "normalize_file",
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


def _resolve_json_output_path(path_or_dir: str | Path, default_filename: str) -> Path:
    """出力パスが .json ファイルでなければディレクトリとみなして default_filename を付与する。"""
    p = Path(path_or_dir)
    if p.suffix.lower() == ".json":
        return p
    return p / default_filename


def normalize_dataset(data: dict) -> tuple[dict, dict]:
    """共通フォーマット JSON 辞書 (sentences: [{original, lines}]) 内の全行を正規化する。

    Returns:
        (正規化後の辞書, 統計情報の辞書)
    """
    normalized_entries: list[dict] = []
    num_sentences = 0

    for item in data.get("sentences", []):
        if not isinstance(item, dict):
            continue
        original = str(item.get("original", ""))
        raw_lines = item.get("lines")
        if isinstance(raw_lines, list):
            norm_lines = [
                normalize_text(str(line))
                for line in raw_lines
                if str(line).strip()
            ]
        elif "sentence" in item:
            # 旧形式互換フォールバック
            sent = str(item.get("sentence", "")).strip()
            norm_lines = [normalize_text(sent)] if sent else []
        else:
            norm_lines = []

        if not norm_lines:
            continue

        normalized_entries.append(
            {
                "original": original,
                "lines": norm_lines,
            }
        )
        num_sentences += len(norm_lines)

    stats = {
        "num_documents": len(normalized_entries),
        "num_sentences": num_sentences,
    }
    return {"sentences": normalized_entries}, stats


def normalize_file(in_path: str | Path, out_path: str | Path) -> dict:
    """共通フォーマット JSON ファイルを読み込んで正規化し、out_path に書き出す。統計辞書を返す。"""
    in_file = Path(in_path)
    out_file = _resolve_json_output_path(out_path, "normalized.json")

    with in_file.open(encoding="utf-8") as f:
        data = json.load(f)

    normalized_data, stats = normalize_dataset(data)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)

    stats["out_normalized_file"] = str(out_file)
    return stats


def run_phase1_preprocess(
    out_slim_file: str | Path,
    out_normalized_file: str | Path,
    out_meta_file: str | Path,
    out_context_file: str | Path,
    *,
    chabsa_directory: str | Path | None = None,
    livedoor_directory: str | Path | None = None,
    aozora_directory: str | Path | None = None,
    input_directory: str | Path | None = None,
    input_context_file: str | Path | None = None,
    out_slim_meta_file: str | Path | None = None,
    out_normalized_meta_file: str | Path | None = None,
) -> dict:
    """chABSA・livedoor・青空文庫（aozora）を共通フォーマット JSON に抽出・統合し、テキスト正規化を実行して metadata.json と context.json を出力する。

    Args:
        out_slim_file: 共通フォーマット抽出後の JSON ファイルパス (またはディレクトリ)
        out_normalized_file: テキスト正規化後の JSON ファイルパス (またはディレクトリ)
        out_meta_file: パイプライン全体のメタデータ出力ファイルパス (JSON)
        out_context_file: 後続フェーズへパスを引き継ぐ context.json の出力ファイルパス
        chabsa_directory: chABSA origin JSON が格納された入力ディレクトリ
        livedoor_directory: livedoor コーパスのテキストが格納された入力ディレクトリ
        aozora_directory: 青空文庫のテキストが格納された入力ディレクトリ (例: samples/aozora)
        input_directory: chabsa_directory の別名（後方互換用）
        input_context_file: 前段の context.json がある場合に入力するファイルパス (任意)
        out_slim_meta_file: 抽出ステップ単体のメタデータ出力ファイルパス (任意)
        out_normalized_meta_file: 正規化ステップ単体のメタデータ出力ファイルパス (任意)

    Returns:
        出力したメタデータの辞書
    """
    resolved_chabsa_dir = (
        Path(chabsa_directory)
        if chabsa_directory is not None
        else (Path(input_directory) if input_directory is not None else None)
    )
    resolved_livedoor_dir = Path(livedoor_directory) if livedoor_directory is not None else None
    resolved_aozora_dir = Path(aozora_directory) if aozora_directory is not None else None

    if (
        resolved_chabsa_dir is None
        and resolved_livedoor_dir is None
        and resolved_aozora_dir is None
    ):
        raise ValueError(
            "chabsa_directory (または input_directory), livedoor_directory, aozora_directory の少なくとも 1 つを指定してください"
        )

    resolved_slim_file = _resolve_json_output_path(out_slim_file, "slim.json")
    resolved_norm_file = _resolve_json_output_path(out_normalized_file, "normalized.json")
    out_meta_file = Path(out_meta_file)
    out_context_file = Path(out_context_file)

    total_started_at = datetime.now(timezone.utc)
    total_start = time.perf_counter()

    # 1. 共通フォーマットへの抽出・統合 (SR1100)
    slim_started_at = datetime.now(timezone.utc)
    slim_start = time.perf_counter()
    slim_stats = build_slim_dataset(
        resolved_slim_file,
        chabsa_dir=resolved_chabsa_dir,
        livedoor_dir=resolved_livedoor_dir,
        aozora_dir=resolved_aozora_dir,
    )
    slim_elapsed = time.perf_counter() - slim_start
    slim_finished_at = datetime.now(timezone.utc)

    slim_meta = {
        "started_at": slim_started_at.isoformat(),
        "finished_at": slim_finished_at.isoformat(),
        "elapsed_seconds": round(slim_elapsed, 3),
        "input": {
            "chabsa_directory": str(resolved_chabsa_dir) if resolved_chabsa_dir else None,
            "livedoor_directory": str(resolved_livedoor_dir) if resolved_livedoor_dir else None,
            "aozora_directory": str(resolved_aozora_dir) if resolved_aozora_dir else None,
        },
        "outputs": {
            "slim_file": str(resolved_slim_file),
        },
        "stats": {
            "num_documents": slim_stats["num_documents"],
            "num_chabsa_documents": slim_stats["num_chabsa_documents"],
            "num_livedoor_documents": slim_stats["num_livedoor_documents"],
            "num_aozora_documents": slim_stats["num_aozora_documents"],
            "num_aozora_skipped_documents": slim_stats["num_aozora_skipped_documents"],
            "num_sentences": slim_stats["num_sentences"],
            "num_chabsa_sentences": slim_stats["num_chabsa_sentences"],
            "num_livedoor_sentences": slim_stats["num_livedoor_sentences"],
            "num_aozora_sentences": slim_stats["num_aozora_sentences"],
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
    norm_stats = normalize_file(resolved_slim_file, resolved_norm_file)
    norm_elapsed = time.perf_counter() - norm_start
    norm_finished_at = datetime.now(timezone.utc)

    norm_meta = {
        "started_at": norm_started_at.isoformat(),
        "finished_at": norm_finished_at.isoformat(),
        "elapsed_seconds": round(norm_elapsed, 3),
        "input": {
            "slim_file": str(resolved_slim_file),
        },
        "outputs": {
            "normalized_file": str(resolved_norm_file),
        },
        "stats": {
            "num_documents": norm_stats["num_documents"],
            "num_sentences": norm_stats["num_sentences"],
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
        "slim_file": str(resolved_slim_file),
        "normalized_file": str(resolved_norm_file),
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
            "chabsa_directory": str(resolved_chabsa_dir) if resolved_chabsa_dir else None,
            "livedoor_directory": str(resolved_livedoor_dir) if resolved_livedoor_dir else None,
            "aozora_directory": str(resolved_aozora_dir) if resolved_aozora_dir else None,
        },
        "outputs": outputs_dict,
        "steps": {
            "slim": slim_meta,
            "normalize": norm_meta,
        },
        "stats": {
            "num_files": norm_stats["num_documents"],
            "num_documents": norm_stats["num_documents"],
            "num_chabsa_documents": slim_stats["num_chabsa_documents"],
            "num_livedoor_documents": slim_stats["num_livedoor_documents"],
            "num_aozora_documents": slim_stats["num_aozora_documents"],
            "num_aozora_skipped_documents": slim_stats["num_aozora_skipped_documents"],
            "num_sentences": norm_stats["num_sentences"],
            "num_chabsa_sentences": slim_stats["num_chabsa_sentences"],
            "num_livedoor_sentences": slim_stats["num_livedoor_sentences"],
            "num_aozora_sentences": slim_stats["num_aozora_sentences"],
        },
    }

    out_meta_file.parent.mkdir(parents=True, exist_ok=True)
    with out_meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # context.json の構築・出力（ファイル・ディレクトリのパスのみを格納し、文章等の実体は持たない）
    context: dict = {}
    if input_context_file is not None:
        context = load_context(input_context_file)

    phase1_context = {
        "chabsa_directory": str(resolved_chabsa_dir) if resolved_chabsa_dir else None,
        "livedoor_directory": str(resolved_livedoor_dir) if resolved_livedoor_dir else None,
        "aozora_directory": str(resolved_aozora_dir) if resolved_aozora_dir else None,
        "slim_file": str(resolved_slim_file),
        "normalized_file": str(resolved_norm_file),
        "meta_file": str(out_meta_file),
    }
    if out_slim_meta_file is not None:
        phase1_context["slim_meta_file"] = str(Path(out_slim_meta_file))
    if out_normalized_meta_file is not None:
        phase1_context["normalized_meta_file"] = str(Path(out_normalized_meta_file))

    context["phase1_preprocess"] = phase1_context
    save_context(context, out_context_file)

    return metadata
