"""`python -m phase3_featurize` で実行するコマンドラインエントリポイント (impl_gen3)。"""

from __future__ import annotations

import argparse
import sys

from . import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EOS_TOKEN,
    DEFAULT_NEWLINE_TOKEN,
    DEFAULT_SEED,
    DEFAULT_SHARD_SIZE,
    run_phase3_featurize,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase3_featurize",
        description=(
            "Phase 2 が出力した context.json から正規化済み共通フォーマット JSON ファイルと採用語彙ファイルを取得し、"
            "100万サンプル単位で重み付き One-Hot 特徴量 X (x_0001.npz, ...) と正解ラベル Y (y_0001.npy, ...) を作成して "
            "フルパス一覧と全体件数を記載した context.json を出力する"
        ),
    )
    parser.add_argument(
        "--input-context-file",
        required=True,
        help="Phase 2 が出力した context.json のファイルパス (例: /tmp/job_gen3_1234/phase2_tokenize/context.json)",
    )
    parser.add_argument(
        "--out-directory",
        default=None,
        help="分割ファイル (x_0001.npz, y_0001.npy, ...) の出力先ディレクトリ (省略時は --out-x-file または --out-meta-file の親ディレクトリ)",
    )
    parser.add_argument(
        "--out-x-file",
        default=None,
        help="出力先指定用の特徴量 X パス (指定された場合はその親ディレクトリに x_0001.npz, ... を出力)",
    )
    parser.add_argument(
        "--out-y-file",
        default=None,
        help="出力先指定用の正解ラベル Y パス (指定された場合はその親ディレクトリに y_0001.npy, ... を出力)",
    )
    parser.add_argument(
        "--out-meta-file",
        required=True,
        help="メタデータ (開始終了時刻、処理時間、入出力パス、パラメータ、統計) の出力ファイル (JSON)",
    )
    parser.add_argument(
        "--out-context-file",
        required=True,
        help="次フェーズへ引き継ぐ context.json の出力ファイルパス (JSON)",
    )

    # 特徴化・分割パラメータ
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="1ファイルあたりの最大サンプル数 (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="文書・サンプルシャッフル用の乱数シード (default: %(default)s)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help="1サンプルあたりの文脈ウィンドウ単語数 (default: %(default)s)",
    )
    parser.add_argument(
        "--bos-token",
        default=DEFAULT_BOS_TOKEN,
        help="文書先頭パディング用の開始記号 (default: %(default)s)",
    )
    parser.add_argument(
        "--eos-token",
        default=DEFAULT_EOS_TOKEN,
        help="文書末尾に付与する終了記号 (default: %(default)s)",
    )
    parser.add_argument(
        "--newline-token",
        default=DEFAULT_NEWLINE_TOKEN,
        help="文書内の文区切り記号 (default: %(default)s)",
    )

    # 任意の上書きオプション (通常は context.json から自動解決)
    parser.add_argument(
        "--normalized-file",
        "--normalized-directory",
        dest="normalized_file",
        default=None,
        help="正規化済み共通フォーマット JSON ファイルを明示的に上書き指定する場合のパス (省略時は context.json から取得)",
    )
    parser.add_argument(
        "--vocab-file",
        default=None,
        help="SentencePiece の .vocab ファイルを明示的に上書き指定する場合のパス (省略時は context.json から取得)",
    )
    parser.add_argument(
        "--model-file",
        default=None,
        help="SentencePiece の .model ファイルを明示的に上書き指定する場合のパス (省略時は context.json から取得)",
    )
    args = parser.parse_args(argv)

    metadata = run_phase3_featurize(
        args.input_context_file,
        args.out_x_file,
        args.out_y_file,
        args.out_meta_file,
        args.out_context_file,
        out_directory=args.out_directory,
        context_length=args.context_length,
        shard_size=args.shard_size,
        seed=args.seed,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        newline_token=args.newline_token,
        normalized_file=args.normalized_file,
        vocab_file=args.vocab_file,
        model_file=args.model_file,
    )

    stats = metadata["stats"]
    print(
        f"phase3_featurize completed: {stats['num_shards']} shards "
        f"(total_samples={stats['num_samples']:,}, vocab_size={stats['vocab_size']:,}, "
        f"nnz={stats['x_nnz']:,}) from {stats['num_documents']:,} docs, "
        f"{stats['num_sentences']:,} sentences ({metadata['elapsed_seconds']}s)"
    )
    print(f"context written: {args.out_context_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 実行例 (job_id=job_gen3_1234)
PYTHONPATH=impl_gen3/src/pipelines/training/phase3_featurize \
python -m phase3_featurize \
  --input-context-file=/tmp/job_gen3_1234/phase2_tokenize/context.json \
  --out-directory=/tmp/job_gen3_1234/phase3_featurize \
  --out-meta-file=/tmp/job_gen3_1234/phase3_featurize/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase3_featurize/context.json \
  --shard-size=1000000 \
  --context-length=32
"""
