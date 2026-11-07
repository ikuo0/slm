"""`python -m weighted_onehot` で実行するコマンドラインエントリポイント (impl_gen3)。"""

from __future__ import annotations

import argparse
import sys

from . import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EOS_TOKEN,
    DEFAULT_NEWLINE_TOKEN,
    build_and_save_xy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weighted_onehot",
        description=(
            "共通フォーマットの normalized JSON と SentencePiece vocab から重み付き One-Hot 特徴量 X (.npz) と "
            "正解トークン ID Y (.npy) を生成する"
        ),
    )
    parser.add_argument(
        "--normalized-file",
        required=True,
        help="正規化済み共通フォーマット JSON ファイル（またはディレクトリ）のパス",
    )
    parser.add_argument(
        "--vocab-file",
        required=True,
        help="SentencePiece の語彙ファイル (.vocab) またはモデルファイル (.model) のパス",
    )
    parser.add_argument(
        "--model-file",
        default=None,
        help="SentencePiece のモデルファイル (.model) のパス (省略時は --vocab-file と同階層の .model を自動検出)",
    )
    parser.add_argument(
        "--out-x-file",
        required=True,
        help="出力する特徴量 X (CSR 疎行列) のファイルパス (.npz)",
    )
    parser.add_argument(
        "--out-y-file",
        required=True,
        help="出力する正解ラベル Y (1次元 int32 配列) のファイルパス (.npy)",
    )
    parser.add_argument(
        "--out-meta-file",
        default=None,
        help="メタデータ (開始終了時刻、処理時間、パラメータ、統計) の出力ファイル (JSON, 任意)",
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
    args = parser.parse_args(argv)

    metadata = build_and_save_xy(
        args.normalized_file,
        args.vocab_file,
        args.out_x_file,
        args.out_y_file,
        out_meta_file=args.out_meta_file,
        model_file=args.model_file,
        context_length=args.context_length,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        newline_token=args.newline_token,
    )

    stats = metadata["stats"]
    print(
        f"weighted_onehot saved: X={ tuple(stats['x_shape']) } (nnz={stats['x_nnz']}), "
        f"Y={ tuple(stats['y_shape']) } from {stats['num_documents']} docs "
        f"({metadata['elapsed_seconds']}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 実行例 (job_id=job_gen3_1234)
PYTHONPATH=impl_gen3/src/libs/weighted_onehot \
python -m weighted_onehot \
  --normalized-file=/tmp/job_gen3_1234/phase1_preprocess/normalized.json \
  --vocab-file=/tmp/job_gen3_1234/phase2_tokenize/vocab2344/sp.vocab \
  --out-x-file=/tmp/job_gen3_1234/phase3_featurize/x.npz \
  --out-y-file=/tmp/job_gen3_1234/phase3_featurize/y.npy \
  --out-meta-file=/tmp/job_gen3_1234/phase3_featurize/metadata.json \
  --context-length=32
"""
