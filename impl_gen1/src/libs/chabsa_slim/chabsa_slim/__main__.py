"""`python -m chabsa_slim` で実行するコマンドラインエントリポイント。"""

import argparse
import sys

from . import slim_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chabsa_slim",
        description="chABSA の JSON から余計なデータを間引いて出力する",
    )
    parser.add_argument(
        "--chabsa-dir",
        required=True,
        help="chABSA の JSON が置かれたディレクトリ (*.json を列挙する)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="出力先ディレクトリ (ファイル名は元のファイル名を踏襲)",
    )
    args = parser.parse_args(argv)

    written = slim_directory(args.chabsa_dir, args.out_dir)
    print(f"{len(written)} files written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
