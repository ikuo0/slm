"""`python -m chabsa_slim` で実行するコマンドラインエントリポイント (impl_gen3)。"""

from __future__ import annotations

import argparse
import sys

from . import build_slim_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chabsa_slim",
        description=(
            "chABSA の JSON および livedoor コーパスの記事テキストから本文行を抽出し、"
            "共通フォーマット JSON (sentences: [{original, lines}]) にまとめて出力する"
        ),
    )
    parser.add_argument(
        "--chabsa-dir",
        default=None,
        help="chABSA の JSON が置かれたディレクトリ (例: samples/chabsa_data/data)",
    )
    parser.add_argument(
        "--livedoor-dir",
        default=None,
        help="livedoor コーパスのテキストが置かれたディレクトリ (例: samples/livedoor/text)",
    )
    parser.add_argument(
        "--out-slim-file",
        "--out-dir",
        dest="out_slim_file",
        required=True,
        help="共通フォーマット JSON の出力先ファイルパス (例: /tmp/job_gen3_1234/phase1_preprocess/slim.json)",
    )
    args = parser.parse_args(argv)

    if not args.chabsa_dir and not args.livedoor_dir:
        parser.error("--chabsa-dir または --livedoor-dir の少なくとも一方を指定してください")

    stats = build_slim_dataset(
        args.out_slim_file,
        chabsa_dir=args.chabsa_dir,
        livedoor_dir=args.livedoor_dir,
    )
    print(
        f"slim written: {stats['out_slim_file']} "
        f"(total={stats['num_documents']} docs / {stats['num_sentences']} lines: "
        f"chabsa={stats['num_chabsa_documents']} docs, "
        f"livedoor={stats['num_livedoor_documents']} docs)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
