"""`python -m text_normalizer` で実行するコマンドラインエントリポイント。"""

import argparse
import sys
from pathlib import Path

from . import normalize_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="text_normalizer",
        description="テキストファイルを正規化して出力する",
    )
    parser.add_argument(
        "--input-text-file",
        required=True,
        help="入力テキストファイル",
    )
    parser.add_argument(
        "--out-normalized-file",
        required=True,
        help="正規化結果の出力ファイル",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input_text_file)
    out_path = Path(args.out_normalized_file)

    text = in_path.read_text(encoding="utf-8")
    normalized = normalize_text(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(normalized, encoding="utf-8")

    print(f"normalized: {in_path} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

