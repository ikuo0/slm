"""`python -m phase2_tokenize` で実行するコマンドラインエントリポイント (impl_gen3)。"""

from __future__ import annotations

import argparse
import sys

from . import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_NEWLINE_TOKEN,
    DEFAULT_SCORE_THRESHOLD,
    run_phase2_tokenize,
)

DEFAULT_SYMBOLS_STR = f"{DEFAULT_BOS_TOKEN},{DEFAULT_EOS_TOKEN},{DEFAULT_NEWLINE_TOKEN}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase2_tokenize",
        description=(
            "Phase 1 が出力した context.json を元にコーパスを作成し、"
            "vocab_size をべき乗で増やしながら SentencePiece 学習・トークン化・レポート作成を行い、"
            "単語数適切指数が最大のものを採用して context.json に出力する"
        ),
    )
    # 入出力指定
    parser.add_argument(
        "--input-context-file",
        required=True,
        help="Phase 1 が出力した context.json のファイルパス (例: /tmp/job_gen3_1234/phase1_preprocess/context.json)",
    )
    parser.add_argument(
        "--out-directory",
        required=True,
        help="出力ベースディレクトリ ({out-directory}/vocab{size}/ に各試行が保存される)",
    )
    parser.add_argument(
        "--out-meta-file",
        required=True,
        help="メタデータ (開始終了時刻、処理時間、探索結果、統計) の出力ファイル (JSON)",
    )
    parser.add_argument(
        "--out-context-file",
        required=True,
        help="Phase 3 へ引き継ぐ context.json の出力ファイルパス (JSON)",
    )
    parser.add_argument(
        "--out-corpus-file",
        default=None,
        help="コーパス（1文1行）の出力ファイルパス (省略時は {out-directory}/corpus.txt)",
    )

    # vocab_size べき乗探索パラメータ
    parser.add_argument(
        "--vocab-size-start",
        type=int,
        required=True,
        help="vocab_size の初期値 (下限割れの場合はスキップしてべき乗で増加)",
    )
    parser.add_argument(
        "--vocab-size-end",
        type=int,
        required=True,
        help="vocab_size の終わり値 (この値を超えた処理を実行するか、上限エラーが出た時点で終了)",
    )
    parser.add_argument(
        "--vocab-size-exponent",
        type=float,
        required=True,
        help="vocab_size を増やすべき乗の指数 (小数指定可、例: 1.03 → ((start^1.03)^1.03)...)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=(
            "単語数適切指数 (appropriateness_score) の終了しきい値 "
            "(この値を超えた時点で探索終了、0 以下で無効、default: %(default)s)"
        ),
    )

    # コーパス・SentencePiece パラメータ
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
        default=DEFAULT_SYMBOLS_STR,
        help="単一トークンとして扱う特殊記号 (カンマ区切り、default: %(default)s)",
    )
    parser.add_argument(
        "--bos-id",
        type=int,
        default=-1,
        help="SentencePiece 組み込み BOS ID (-1 で予約解除し user_defined_symbols の <s> を使う) (default: %(default)s)",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=-1,
        help="SentencePiece 組み込み EOS ID (-1 で予約解除し user_defined_symbols の </s> を使う) (default: %(default)s)",
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

    metadata = run_phase2_tokenize(
        args.input_context_file,
        args.out_directory,
        args.out_meta_file,
        args.out_context_file,
        out_corpus_file=args.out_corpus_file,
        vocab_size_start=args.vocab_size_start,
        vocab_size_end=args.vocab_size_end,
        vocab_size_exponent=args.vocab_size_exponent,
        score_threshold=args.score_threshold,
        newline_token=args.newline_token,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        user_defined_symbols=user_defined_symbols,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        output_format=args.output_format,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )

    search = metadata["search_result"]
    print(
        f"phase2_tokenize completed: {search['num_trials']} trials, "
        f"{search['num_skipped']} skipped "
        f"(stopped: {search['stopped_reason']}, {metadata['elapsed_seconds']}s)"
    )
    for s in search["skipped_trials"]:
        print(f"  - vocab_size={s['vocab_size']}: skipped ({s['reason']})")
    for t in search["trials"]:
        marker = "*" if t["vocab_size"] == search["selected_vocab_size"] else " "
        print(
            f"  {marker} vocab_size={t['vocab_size']}: "
            f"score={t['appropriateness_score']:.6f}, "
            f"chars/token={t['chars_per_token']:.3f}, "
            f"freq_ratio={t['frequent_vocab_ratio']:.3f}, "
            f"entropy={t['normalized_entropy']:.3f}"
        )
    print(
        f"selected vocab_size={search['selected_vocab_size']} "
        f"(score={search['selected_appropriateness_score']:.6f})"
    )
    print(f"context written: {args.out_context_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 実行例 (job_id=job_gen3_1234)
PYTHONPATH=impl_gen3/src/pipelines/training/phase2_tokenize \
python -m phase2_tokenize \
  --input-context-file=/tmp/job_gen3_1234/phase1_preprocess/context.json \
  --out-directory=/tmp/job_gen3_1234/phase2_tokenize \
  --out-meta-file=/tmp/job_gen3_1234/phase2_tokenize/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase2_tokenize/context.json \
  --vocab-size-start=1500 \
  --vocab-size-end=20000 \
  --vocab-size-exponent=1.03 \
  --score-threshold=0.70 \
  --model-type=unigram \
  --character-coverage=0.9995 \
  --user-defined-symbols='<s>,</s>,<n>' \
  --output-format=piece
"""
