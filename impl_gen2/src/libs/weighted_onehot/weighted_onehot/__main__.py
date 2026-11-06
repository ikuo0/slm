"""`python -m weighted_onehot` で実行するコマンドラインエントリポイント (impl_gen2)。"""

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
    build_and_save_xy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weighted_onehot",
        description=(
            "normalized JSON と SentencePiece vocab から、3つのハッシュインデックス "
            "(hash1(word)%vector_dim, hash2(word)%vector_dim, hash3(word)%vector_dim) で圧縮した "
            "重み付き One-Hot 特徴量 X (.npz) と正解トークン ID Y (.npy) を生成する"
        ),
    )
    parser.add_argument(
        "--normalized-file",
        required=True,
        help="正規化済み simplified JSON ファイル、または JSON が格納されたディレクトリのパス",
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
        help="出力する圧縮特徴量 X (CSR 疎行列, shape=(N, vector_dim)) のファイルパス (.npz)",
    )
    parser.add_argument(
        "--out-y-file",
        required=True,
        help="出力する正解ラベル Y (1次元 int32 配列, shape=(N,)) のファイルパス (.npy)",
    )
    parser.add_argument(
        "--out-meta-file",
        default=None,
        help="メタデータ (開始終了時刻、処理時間、パラメータ、統計) の出力ファイル (JSON, 任意)",
    )
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
        help="1単語あたりに使用するハッシュインデックスの個数 (default: %(default)s)",
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
        vector_dim=args.vector_dim,
        num_hashes=args.num_hashes,
        context_length=args.context_length,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        newline_token=args.newline_token,
    )

    stats = metadata["stats"]
    print(
        f"weighted_onehot saved: X={tuple(stats['x_shape'])} "
        f"(vector_dim={stats['vector_dim']}, num_hashes={stats['num_hashes']}, "
        f"unique_tuples={stats['unique_word_hash_tuples']}/{stats['vocab_size']}, nnz={stats['x_nnz']}), "
        f"Y={tuple(stats['y_shape'])} from {stats['num_documents']} docs "
        f"({metadata['elapsed_seconds']}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 実行例 (job_id=job1234)
PYTHONPATH=impl_gen2/src/libs/weighted_onehot \
python -m weighted_onehot \
  --normalized-file=/tmp/job1234/phase1_preprocess/normalized \
  --vocab-file=/tmp/job1234/phase2_tokenize/vocab4808/sp.vocab \
  --out-x-file=/tmp/job1234_gen2/phase3_featurize/x.npz \
  --out-y-file=/tmp/job1234_gen2/phase3_featurize/y.npy \
  --out-meta-file=/tmp/job1234_gen2/phase3_featurize/metadata.json \
  --vector-dim=256 \
  --num-hashes=3 \
  --context-length=32
"""
