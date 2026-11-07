"""`python -m phase5_inference` で実行するコマンドラインエントリポイント。"""

from __future__ import annotations

import argparse
import sys

from . import (
    DEFAULT_MAX_CHARS,
    DEFAULT_NUM_DOCUMENTS,
    DEFAULT_PROB_POWER,
    DEFAULT_SAMPLING_MODE,
    run_phase5_inference,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase5_inference",
        description=(
            "Phase 4 が出力した context.json から学習済みモデルと語彙を取得し、"
            "指定本数の文章を生成して generated_texts.json, 目視用テキスト, metadata.json, context.json を出力する"
        ),
    )
    # 入出力指定 (すべて個別ファイルパスで指定)
    parser.add_argument(
        "--input-context-file",
        required=True,
        help="Phase 4 が出力した context.json のファイルパス (例: /tmp/job_gen3_1234/phase4_training/context.json)",
    )
    parser.add_argument(
        "--out-texts-file",
        required=True,
        help="生成した文章一覧の出力ファイルパス (JSON, 例: /tmp/job_gen3_1234/phase5_inference/generated_texts.json)",
    )
    parser.add_argument(
        "--out-readable-file",
        "--out-text-file",
        dest="out_readable_file",
        required=True,
        help="目視確認用テキストの出力ファイルパス (例: /tmp/job_gen3_1234/phase5_inference/generated_texts.txt)",
    )
    parser.add_argument(
        "--out-meta-file",
        required=True,
        help="メタデータの出力ファイルパス (JSON, 例: /tmp/job_gen3_1234/phase5_inference/metadata.json)",
    )
    parser.add_argument(
        "--out-context-file",
        required=True,
        help="context.json の出力ファイルパス (JSON, 例: /tmp/job_gen3_1234/phase5_inference/context.json)",
    )

    # 生成制御オプション (4つ + seed)
    parser.add_argument(
        "--num-documents",
        type=int,
        default=DEFAULT_NUM_DOCUMENTS,
        help="生成する文章（文書）の本数 (default: %(default)s)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="1文書あたりの最大文字数 (</s> が現れなくても打ち切る上限、default: %(default)s)",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "greedy"],
        default=DEFAULT_SAMPLING_MODE,
        help="単語選択方式: random=確率抽選, greedy=最大確率を取り続ける (default: %(default)s)",
    )
    parser.add_argument(
        "--prob-power",
        type=float,
        default=DEFAULT_PROB_POWER,
        help="random モード時に確率を何乗してから正規化するか (1.0=そのまま, 2.0=二乗して高確率に寄せる) (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: %(default)s)",
    )

    # 任意の上書きオプション (通常は context.json から自動解決)
    parser.add_argument(
        "--nn-model-file",
        default=None,
        help="学習済み NN モデル (.keras) を明示的に上書き指定する場合のパス (任意)",
    )
    parser.add_argument(
        "--vocab-file",
        default=None,
        help="SentencePiece の .vocab ファイルを明示的に上書き指定する場合のパス (任意)",
    )
    parser.add_argument(
        "--sp-model-file",
        default=None,
        help="SentencePiece の .model ファイルを明示的に上書き指定する場合のパス (任意)",
    )
    args = parser.parse_args(argv)

    metadata = run_phase5_inference(
        args.input_context_file,
        args.out_texts_file,
        args.out_meta_file,
        args.out_context_file,
        out_readable_file=args.out_readable_file,
        num_documents=args.num_documents,
        max_chars=args.max_chars,
        sampling_mode=args.sampling_mode,
        prob_power=args.prob_power,
        seed=args.seed,
        nn_model_file=args.nn_model_file,
        vocab_file=args.vocab_file,
        sp_model_file=args.sp_model_file,
    )

    stats = metadata["stats"]
    print(
        f"phase5_inference completed: {stats['num_documents']} docs generated "
        f"(eos={stats['eos_stopped_count']}, max_chars={stats['max_chars_stopped_count']}, "
        f"avg_chars={stats['avg_chars']}, avg_sentences={stats['avg_sentences']}, "
        f"peak_rss={stats['peak_rss_mb']}MB, {metadata['elapsed_seconds']}s)"
    )
    print(f"texts written: {args.out_texts_file}")
    print(f"readable text written: {args.out_readable_file}")
    print(f"context written: {args.out_context_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 実行例1: 確率を二乗 (--prob-power=2.0) して高確率に寄せたランダム生成で 10 文書作成
PYTHONPATH=impl_gen3/src/pipelines/training/phase5_inference \
python -m phase5_inference \
  --input-context-file=/tmp/job_gen3_1234/phase4_training/context.json \
  --out-texts-file=/tmp/job_gen3_1234/phase5_inference/generated_texts.json \
  --out-readable-file=/tmp/job_gen3_1234/phase5_inference/generated_texts.txt \
  --out-meta-file=/tmp/job_gen3_1234/phase5_inference/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase5_inference/context.json \
  --num-documents=10 \
  --max-chars=1000 \
  --sampling-mode=random \
  --prob-power=2.0 \
  --seed=555

# 実行例2: 素の確率ランダム (--prob-power=1.0)
PYTHONPATH=impl_gen3/src/pipelines/training/phase5_inference \
python -m phase5_inference \
  --input-context-file=/tmp/job_gen3_1234/phase4_training/context.json \
  --out-texts-file=/tmp/job_gen3_1234/phase5_inference/generated_texts_p1.json \
  --out-readable-file=/tmp/job_gen3_1234/phase5_inference/generated_texts_p1.txt \
  --out-meta-file=/tmp/job_gen3_1234/phase5_inference/metadata_p1.json \
  --out-context-file=/tmp/job_gen3_1234/phase5_inference/context.json \
  --num-documents=10 \
  --max-chars=1000 \
  --sampling-mode=random \
  --prob-power=1.0 \
  --seed=42

# 実行例3: 最大確率を取り続ける (--sampling-mode=greedy) で代表 1 文書を作成
PYTHONPATH=impl_gen3/src/pipelines/training/phase5_inference \
python -m phase5_inference \
  --input-context-file=/tmp/job_gen3_1234/phase4_training/context.json \
  --out-texts-file=/tmp/job_gen3_1234/phase5_inference/generated_greedy.json \
  --out-readable-file=/tmp/job_gen3_1234/phase5_inference/generated_greedy.txt \
  --out-meta-file=/tmp/job_gen3_1234/phase5_inference/metadata_greedy.json \
  --out-context-file=/tmp/job_gen3_1234/phase5_inference/context.json \
  --num-documents=1 \
  --max-chars=1000 \
  --sampling-mode=greedy \
  --seed=42
"""
