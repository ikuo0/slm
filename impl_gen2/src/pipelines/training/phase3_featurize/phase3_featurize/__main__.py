"""`python -m phase3_featurize` で実行するコマンドラインエントリポイント (impl_gen2)。"""

from __future__ import annotations

import argparse
import sys

from . import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_EOS_TOKEN,
    DEFAULT_NEWLINE_TOKEN,
    DEFAULT_NUM_HASHES,
    DEFAULT_VECTOR_DIM,
    run_phase3_featurize,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase3_featurize",
        description=(
            "Phase 2 が出力した context.json から正規化済み JSON ディレクトリと採用語彙ファイルを取得し、"
            "3つのハッシュインデックス (hash1(word)%vector_dim, hash2(word)%vector_dim, hash3(word)%vector_dim) "
            "で圧縮した重み付き One-Hot 特徴量 X (.npz) と正解ラベル Y (.npy) を作成して context.json を出力する"
        ),
    )
    # 入出力指定 (出力はすべて個別のファイルパスで指定)
    parser.add_argument(
        "--input-context-file",
        required=True,
        help="Phase 2 が出力した context.json のファイルパス (例: /tmp/job_gen2_1234/phase2_tokenize/context.json)",
    )
    parser.add_argument(
        "--out-x-file",
        required=True,
        help="出力する圧縮特徴量 X (CSR 疎行列, shape=(N, vector_dim)) のファイルパス (.npz)",
    )
    parser.add_argument(
        "--out-y-file",
        required=True,
        help="出力する正解ラベル Y (1次元 int32 配列, shape=(N,)) のファイルパス (.npy)",
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

    # 特徴化パラメータ
    parser.add_argument(
        "--vector-dim",
        "--compress-dim",
        dest="vector_dim",
        type=int,
        default=DEFAULT_VECTOR_DIM,
        help="X データを圧縮するベクトル次元数 (`hash_k(word) %% vector_dim` をインデックスにする, default: %(default)s)",
    )
    parser.add_argument(
        "--num-hashes",
        type=int,
        default=DEFAULT_NUM_HASHES,
        help="1単語あたりに使用するハッシュインデックス数 (default: %(default)s)",
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
        "--normalized-directory",
        default=None,
        help="正規化済み JSON ディレクトリを明示的に上書き指定する場合のパス (省略時は context.json から取得)",
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
        vector_dim=args.vector_dim,
        num_hashes=args.num_hashes,
        context_length=args.context_length,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        newline_token=args.newline_token,
        normalized_directory=args.normalized_directory,
        vocab_file=args.vocab_file,
        model_file=args.model_file,
    )

    stats = metadata["stats"]
    print(
        f"phase3_featurize completed: X={tuple(stats['x_shape'])} "
        f"(vector_dim={stats['vector_dim']}, num_hashes={stats['num_hashes']}, "
        f"unique_tuples={stats['unique_word_hash_tuples']}/{stats['vocab_size']}, nnz={stats['x_nnz']}), "
        f"Y={tuple(stats['y_shape'])} from {stats['num_documents']} docs, "
        f"{stats['num_sentences']} sentences ({metadata['elapsed_seconds']}s)"
    )
    print(f"context written: {args.out_context_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 実行例 (job_id=job_gen2_1234)
PYTHONPATH=impl_gen2/src/pipelines/training/phase3_featurize \
python -m phase3_featurize \
  --input-context-file=/tmp/job_gen2_1234/phase2_tokenize/context.json \
  --out-x-file=/tmp/job_gen2_1234/phase3_featurize/x.npz \
  --out-y-file=/tmp/job_gen2_1234/phase3_featurize/y.npy \
  --out-meta-file=/tmp/job_gen2_1234/phase3_featurize/metadata.json \
  --out-context-file=/tmp/job_gen2_1234/phase3_featurize/context.json \
  --vector-dim=256 \
  --num-hashes=3 \
  --context-length=32
"""
