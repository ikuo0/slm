"""`python -m chabsa_slim` で実行するコマンドラインエントリポイント (impl_gen3)。"""

from __future__ import annotations

import argparse
import sys

from . import build_slim_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chabsa_slim",
        description=(
            "chABSA の JSON、livedoor コーパスの記事テキスト、および青空文庫テキストから本文行を抽出し、"
            "共通フォーマット JSON (sentences: [{original, lines}]) に UTF-8 でまとめて出力する"
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
        "--aozora-dir",
        default=None,
        help="青空文庫のテキストが置かれたディレクトリ (例: samples/aozora)",
    )
    parser.add_argument(
        "--out-slim-file",
        "--out-dir",
        dest="out_slim_file",
        required=True,
        help="共通フォーマット JSON の出力先ファイルパス (例: /tmp/job_gen3_1234/phase1_preprocess/slim.json)",
    )
    args = parser.parse_args(argv)

    if not args.chabsa_dir and not args.livedoor_dir and not args.aozora_dir:
        parser.error(
            "--chabsa-dir, --livedoor-dir, --aozora-dir の少なくとも 1 つを指定してください"
        )

    stats = build_slim_dataset(
        args.out_slim_file,
        chabsa_dir=args.chabsa_dir,
        livedoor_dir=args.livedoor_dir,
        aozora_dir=args.aozora_dir,
    )
    print(
        f"slim written: {stats['out_slim_file']} "
        f"(total={stats['num_documents']} docs / {stats['num_sentences']} lines: "
        f"chabsa={stats['num_chabsa_documents']} docs, "
        f"livedoor={stats['num_livedoor_documents']} docs, "
        f"aozora={stats['num_aozora_documents']} docs [skipped={stats['num_aozora_skipped_documents']}])"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
