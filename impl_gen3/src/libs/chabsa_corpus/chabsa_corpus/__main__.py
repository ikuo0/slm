"""`python -m chabsa_corpus` で実行するコマンドラインエントリポイント (impl_gen3)。"""

import argparse
import sys

from . import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_NEWLINE_TOKEN,
    build_corpus,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chabsa_corpus",
        description=(
            "共通フォーマット JSON ファイル（またはディレクトリ）から SentencePiece 用コーパスを生成する "
            "(1文書を <s>...</s> で囲み、<n>\\n 区切りで 1文 = 1行 として出力)"
        ),
    )
    parser.add_argument(
        "--json-file",
        "--json-dir",
        dest="json_path",
        required=True,
        help="共通フォーマット JSON ファイル、または JSON が置かれたディレクトリのパス",
    )
    parser.add_argument(
        "--out-corpus-file",
        required=True,
        help="コーパスの出力ファイル",
    )
    parser.add_argument(
        "--newline-token",
        default=DEFAULT_NEWLINE_TOKEN,
        help="文書内の文区切り (改行相当) の特殊記号 (default: %(default)s)",
    )
    parser.add_argument(
        "--bos-token",
        default=DEFAULT_BOS_TOKEN,
        help="文書先頭に付与する開始記号 (default: %(default)s)",
    )
    parser.add_argument(
        "--eos-token",
        default=DEFAULT_EOS_TOKEN,
        help="文書末尾に付与する終了記号 (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    stats = build_corpus(
        args.json_path,
        args.out_corpus_file,
        newline_token=args.newline_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
    )

    print(
        f"corpus written: {args.out_corpus_file} "
        f"({stats['num_documents']} documents, {stats['num_sentences']} sentences)"
    )
    print(f"special tokens for SentencePiece user_defined_symbols: {stats['special_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
