"""`python -m phase1_preprocess` で実行するコマンドラインエントリポイント。"""

from __future__ import annotations

import argparse
import sys

from . import run_phase1_preprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase1_preprocess",
        description="chABSA データの簡素化およびテキスト正規化を実行し、metadata.json と context.json を出力する",
    )
    parser.add_argument(
        "--input-directory",
        required=True,
        help="入力データのディレクトリ (例: impl_gen2/samples/chabsa_data/origin)",
    )
    parser.add_argument(
        "--out-slim-directory",
        required=True,
        help="データ簡素化済み JSON の出力先ディレクトリ (例: /tmp/job1234/phase1_preprocess/slim)",
    )
    parser.add_argument(
        "--out-normalized-directory",
        required=True,
        help="データ正規化済み JSON の出力先ディレクトリ (例: /tmp/job1234/phase1_preprocess/normalized)",
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
        help="データ簡素化ステップ単体のメタデータ出力ファイル (JSON, 任意)",
    )
    parser.add_argument(
        "--out-normalized-meta-file",
        default=None,
        help="データ正規化ステップ単体のメタデータ出力ファイル (JSON, 任意)",
    )
    args = parser.parse_args(argv)

    metadata = run_phase1_preprocess(
        args.input_directory,
        args.out_slim_directory,
        args.out_normalized_directory,
        args.out_meta_file,
        args.out_context_file,
        input_context_file=args.input_context_file,
        out_slim_meta_file=args.out_slim_meta_file,
        out_normalized_meta_file=args.out_normalized_meta_file,
    )

    stats = metadata["stats"]
    print(
        f"phase1_preprocess completed: {stats['num_files']} files, "
        f"{stats['num_sentences']} sentences ({metadata['elapsed_seconds']}s)"
    )
    print(f"context written: {args.out_context_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 実行例 (job_id=job1234)
PYTHONPATH=impl_gen2/src/pipelines/training/phase1_preprocess \
python -m phase1_preprocess \
  --input-directory=impl_gen2/samples/chabsa_data/origin \
  --out-slim-directory=/tmp/job1234/phase1_preprocess/slim \
  --out-normalized-directory=/tmp/job1234/phase1_preprocess/normalized \
  --out-meta-file=/tmp/job1234/phase1_preprocess/metadata.json \
  --out-context-file=/tmp/job1234/phase1_preprocess/context.json
"""

