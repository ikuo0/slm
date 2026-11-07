"""`python -m sp_tokenizer` で実行するコマンドラインエントリポイント。"""

import argparse
import sys

from . import DEFAULT_USER_DEFINED_SYMBOLS, train_and_tokenize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sp_tokenizer",
        description="SentencePiece モデルを学習してテキストを1行ずつトークン化する",
    )
    # 入出力 (出力はすべて個別のファイルパスで指定する)
    parser.add_argument(
        "--input-text-file",
        required=True,
        help="入力テキストファイル (1行1文、文書単位で <s>...</s>・文区切り <n>)",
    )
    parser.add_argument(
        "--out-model-file",
        required=True,
        help="学習済み SentencePiece モデルの出力ファイル (.model)",
    )
    parser.add_argument(
        "--out-vocab-file",
        required=True,
        help="語彙の出力ファイル (.vocab)",
    )
    parser.add_argument(
        "--out-tokens-file",
        required=True,
        help="トークン化結果の出力ファイル (1行1文、空白区切り)",
    )
    parser.add_argument(
        "--out-meta-file",
        required=True,
        help="メタデータ (開始終了時間、処理時間、SentencePiece パラメータ) の出力ファイル (JSON)",
    )
    # SentencePiece オプション
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="語彙サイズ (トークン数) (default: %(default)s)",
    )
    parser.add_argument(
        "--model-type",
        choices=["unigram", "bpe", "char", "word"],
        default="unigram",
        help="SentencePiece のモデル種別 (default: %(default)s)",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="文字カバレッジ (default: %(default)s)",
    )
    parser.add_argument(
        "--user-defined-symbols",
        default=",".join(DEFAULT_USER_DEFINED_SYMBOLS),
        help="単一トークンとして扱う特殊記号 (カンマ区切り、default: %(default)s)",
    )
    parser.add_argument(
        "--bos-id",
        type=int,
        default=-1,
        help="SentencePiece 組み込み BOS の ID (-1 で無効化し <s> を user_defined_symbols として扱う) (default: %(default)s)",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=-1,
        help="SentencePiece 組み込み EOS の ID (-1 で無効化し </s> を user_defined_symbols として扱う) (default: %(default)s)",
    )
    parser.add_argument(
        "--output-format",
        choices=["piece", "id"],
        default="piece",
        help="トークン出力形式: piece=トークン文字列, id=トークンID (default: %(default)s)",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="各行の先頭に組み込み BOS を付与する (bos-id >= 0 の場合のみ有効)",
    )
    parser.add_argument(
        "--add-eos",
        action="store_true",
        help="各行の末尾に組み込み EOS を付与する (eos-id >= 0 の場合のみ有効)",
    )
    args = parser.parse_args(argv)

    user_defined_symbols = [s for s in args.user_defined_symbols.split(",") if s]

    metadata = train_and_tokenize(
        args.input_text_file,
        args.out_model_file,
        args.out_vocab_file,
        args.out_tokens_file,
        args.out_meta_file,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        user_defined_symbols=user_defined_symbols,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        output_format=args.output_format,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )

    stats = metadata["stats"]
    print(
        f"tokenized: {stats['num_lines']} lines, {stats['num_tokens']} tokens "
        f"({metadata['elapsed_seconds']}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 学習+トークン化（コーパス内に <s>, </s>, <n> が埋め込まれている前提）
python -m sp_tokenizer \
  --input-text-file=corpus.txt ... \
  --user-defined-symbols='<s>,</s>,<n>'
"""
