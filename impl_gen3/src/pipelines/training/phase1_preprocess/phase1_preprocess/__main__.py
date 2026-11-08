"""`python -m phase1_preprocess` で実行するコマンドラインエントリポイント (impl_gen3)。"""

from __future__ import annotations

import argparse
import sys

from . import run_phase1_preprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase1_preprocess",
        description=(
            "chABSA データ、livedoor コーパス、および青空文庫テキストを共通フォーマット JSON に抽出・統合し、"
            "テキスト正規化を実行して metadata.json と context.json を UTF-8 で出力する"
        ),
    )
    parser.add_argument(
        "--chabsa-directory",
        "--input-directory",
        dest="chabsa_directory",
        default=None,
        help="chABSA データのディレクトリ (例: samples/chabsa_data/data)",
    )
    parser.add_argument(
        "--livedoor-directory",
        dest="livedoor_directory",
        default=None,
        help="livedoor コーパスのテキストディレクトリ (例: samples/livedoor/text)",
    )
    parser.add_argument(
        "--aozora-directory",
        dest="aozora_directory",
        default=None,
        help="青空文庫のテキストディレクトリ (例: samples/aozora)",
    )
    parser.add_argument(
        "--out-slim-file",
        "--out-slim-directory",
        dest="out_slim_file",
        required=True,
        help="共通フォーマット抽出済み JSON の出力先ファイルパス (例: /tmp/job_gen3_1234/phase1_preprocess/slim.json)",
    )
    parser.add_argument(
        "--out-normalized-file",
        "--out-normalized-directory",
        dest="out_normalized_file",
        required=True,
        help="データ正規化済み JSON の出力先ファイルパス (例: /tmp/job_gen3_1234/phase1_preprocess/normalized.json)",
    )
    parser.add_argument(
        "--out-meta-file",
        required=True,
        help="メタデータ (開始終了時刻、処理時間、入出力パス、統計) の出力ファイル (JSON)",
    )
    parser.add_argument(
        "--out-context-file",
        required=True,
        help="後続フェーズへ引き継ぐ context.json の出力ファイルパス (JSON)",
    )
    parser.add_argument(
        "--input-context-file",
        default=None,
        help="前段の context.json がある場合の入力ファイルパス (JSON, 任意)",
    )
    parser.add_argument(
        "--out-slim-meta-file",
        default=None,
        help="共通フォーマット抽出ステップ単体のメタデータ出力ファイル (JSON, 任意)",
    )
    parser.add_argument(
        "--out-normalized-meta-file",
        default=None,
        help="データ正規化ステップ単体のメタデータ出力ファイル (JSON, 任意)",
    )
    args = parser.parse_args(argv)

    if (
        not args.chabsa_directory
        and not args.livedoor_directory
        and not args.aozora_directory
    ):
        parser.error(
            "--chabsa-directory (または --input-directory), --livedoor-directory, --aozora-directory の少なくとも 1 つを指定してください"
        )

    metadata = run_phase1_preprocess(
        args.out_slim_file,
        args.out_normalized_file,
        args.out_meta_file,
        args.out_context_file,
        chabsa_directory=args.chabsa_directory,
        livedoor_directory=args.livedoor_directory,
        aozora_directory=args.aozora_directory,
        input_context_file=args.input_context_file,
        out_slim_meta_file=args.out_slim_meta_file,
        out_normalized_meta_file=args.out_normalized_meta_file,
    )

    stats = metadata["stats"]
    print(
        f"phase1_preprocess completed: {stats['num_documents']} docs "
        f"(chabsa={stats['num_chabsa_documents']}, "
        f"livedoor={stats['num_livedoor_documents']}, "
        f"aozora={stats['num_aozora_documents']} [skipped={stats['num_aozora_skipped_documents']}]), "
        f"{stats['num_sentences']} lines ({metadata['elapsed_seconds']}s)"
    )
    print(f"context written: {args.out_context_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 実行例 (job_id=job_gen3_1234)
PYTHONPATH=impl_gen3/src/pipelines/training/phase1_preprocess \
python -m phase1_preprocess \
  --chabsa-directory=samples/chabsa_data/data \
  --livedoor-directory=samples/livedoor/text \
  --aozora-directory=samples/aozora \
  --out-slim-file=/tmp/job_gen3_1234/phase1_preprocess/slim.json \
  --out-normalized-file=/tmp/job_gen3_1234/phase1_preprocess/normalized.json \
  --out-meta-file=/tmp/job_gen3_1234/phase1_preprocess/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase1_preprocess/context.json
"""
