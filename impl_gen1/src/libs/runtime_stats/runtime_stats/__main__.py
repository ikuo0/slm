"""`python -m runtime_stats` で現在のプロセスのメモリ統計を表示・JSON出力するエントリポイント。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import get_memory_stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="runtime_stats",
        description="/proc/self/status からプロセスのメモリ使用量統計 (VmHWM / VmRSS / VmPeak) を取得する",
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="メモリ統計 JSON の出力先ファイルパス (省略時は標準出力のみ)",
    )
    args = parser.parse_args(argv)

    stats = get_memory_stats()
    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
