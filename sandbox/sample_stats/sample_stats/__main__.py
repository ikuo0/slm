"""chABSA・livedoor オリジナルデータ統計集計 CLI エントリポイント。

実行例:
    PYTHONPATH=sandbox/sample_stats python -m sample_stats
    PYTHONPATH=sandbox/sample_stats python -m sample_stats --corpus=chabsa
    PYTHONPATH=sandbox/sample_stats python -m sample_stats --corpus=livedoor
    PYTHONPATH=sandbox/sample_stats python -m sample_stats --out-json=/tmp/sample_stats.json --out-md=/tmp/sample_stats.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import analyze_corpora, format_summary_markdown

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHABSA_DIR = _REPO_ROOT / "samples" / "chabsa_data" / "data"
DEFAULT_LIVEDOOR_DIR = _REPO_ROOT / "samples" / "livedoor" / "text"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 引数をパースする。"""
    parser = argparse.ArgumentParser(
        prog="sample_stats",
        description="chABSA および livedoor コーパスのオリジナル状態（正規化前・SentencePiece前）の文章統計を集計する",
    )
    parser.add_argument(
        "--corpus",
        choices=("all", "chabsa", "livedoor"),
        default="all",
        help="集計対象コーパス (all / chabsa / livedoor, default: all)",
    )
    parser.add_argument(
        "--chabsa-directory",
        type=Path,
        default=DEFAULT_CHABSA_DIR,
        help=f"chABSA オリジナル JSON ディレクトリ (default: {DEFAULT_CHABSA_DIR})",
    )
    parser.add_argument(
        "--livedoor-directory",
        type=Path,
        default=DEFAULT_LIVEDOOR_DIR,
        help=f"livedoor オリジナル text ディレクトリ (default: {DEFAULT_LIVEDOOR_DIR})",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="統計結果を JSON ファイルとして保存するパス (任意)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="統計レポートを Markdown ファイルとして保存するパス (任意)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI メイン処理。"""
    args = parse_args(argv)

    chabsa_dir = args.chabsa_directory if args.corpus in ("all", "chabsa") else None
    livedoor_dir = args.livedoor_directory if args.corpus in ("all", "livedoor") else None

    t0 = time.perf_counter()
    report = analyze_corpora(
        chabsa_dir=chabsa_dir,
        livedoor_dir=livedoor_dir,
    )
    elapsed = time.perf_counter() - t0
    report["elapsed_seconds"] = round(elapsed, 3)

    md_text = format_summary_markdown(report)
    print(md_text)
    print(f"(集計処理時間: {elapsed:.3f} 秒)")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"JSON 統計出力完了: {args.out_json}")

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md_text + "\n", encoding="utf-8")
        print(f"Markdown レポート出力完了: {args.out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
