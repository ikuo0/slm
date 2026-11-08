"""`python -m phase4_training` で実行するコマンドラインエントリポイント。"""

from __future__ import annotations

import argparse
import sys

from . import (
    DEFAULT_ACTIVATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_UNITS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    run_phase4_training,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="phase4_training",
        description=(
            "Phase 3 が出力した context.json から X (.npz) と Y (.npy) を取得し、"
            "TensorFlow で NN モデルを学習して metadata.json と context.json を出力する "
            "(チャンクごとに保存・再開、全チャンク完了後に収束判定)"
        ),
    )
    parser.add_argument(
        "--work-directory",
        required=True,
        help="作業ディレクトリ (model.keras, metadata.json, context.json が保存される)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="中断再開フラグ (指定時は device、batch-size と出力メタデータ指定以外の引数を無視し、作業ディレクトリの metadata.json から設定を読み込んで再開する)",
    )
    parser.add_argument(
        "--input-context-file",
        default=None,
        help="Phase 3 が出力した context.json のファイルパス (新規学習時は必須)",
    )
    parser.add_argument(
        "--out-model-file",
        default=None,
        help="学習済みモデルの出力ファイルパス (省略時は {work-directory}/model.keras)",
    )
    parser.add_argument(
        "--out-meta-file",
        default=None,
        help="メタデータの出力ファイルパス (省略時は {work-directory}/metadata.json)",
    )
    parser.add_argument(
        "--out-context-file",
        default=None,
        help="次フェーズへ引き継ぐ context.json の出力ファイルパス (省略時は {work-directory}/context.json)",
    )

    # 学習・収束パラメータ
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="最大エポック数 (上限回数、default: %(default)s)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help="収束判定の最小 Loss 改善幅 (0 以下で無効、default: %(default)s)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Loss 改善幅 < min-delta が何エポック連続したら収束終了とするか (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"ミニバッチサイズ。再開時も変更可能。省略時は新規{DEFAULT_BATCH_SIZE}、再開時は保存値",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="学習率 (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-units",
        default=",".join(str(u) for u in DEFAULT_HIDDEN_UNITS),
        help="隠れ層のニューロン数 (カンマ区切り、default: %(default)s)",
    )
    parser.add_argument(
        "--activation",
        default=DEFAULT_ACTIVATION,
        help="隠れ層の活性化関数 (default: %(default)s)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="ドロップアウト率 (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: %(default)s)",
    )

    # 任意の上書きオプション (通常は context.json から自動解決)
    parser.add_argument(
        "--x-file",
        default=None,
        help="特徴量 X (.npz) を明示的に上書き指定する場合のパス (省略時は context.json から取得)",
    )
    parser.add_argument(
        "--y-file",
        default=None,
        help="正解ラベル Y (.npy) を明示的に上書き指定する場合のパス (省略時は context.json から取得)",
    )
    parser.add_argument(
        "--input-mode", choices=("dense", "sparse"), default="dense",
        help="先頭層の計算方式: dense=密展開、sparse=有効インデックスから直接計算 (default: %(default)s)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "gpu"), default="auto",
        help="実行デバイス: auto=GPU優先、cpu=CPUのみ、gpu=GPU必須。再開時も指定可能 (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if not args.resume:
        if not args.input_context_file and not (args.x_file and args.y_file):
            parser.error(
                "新規学習時 (--resume 未指定時) は --input-context-file (または --x-file と --y-file) の指定が必須です"
            )

    hidden_units = tuple(int(u) for u in args.hidden_units.split(",") if u.strip())

    metadata = run_phase4_training(
        args.work_directory,
        resume=args.resume,
        input_context_file=args.input_context_file,
        out_model_file=args.out_model_file,
        out_meta_file=args.out_meta_file,
        out_context_file=args.out_context_file,
        epochs=args.epochs,
        min_delta=args.min_delta,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_units=hidden_units,
        input_mode=args.input_mode,
        device=args.device,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        seed=args.seed,
        x_file=args.x_file,
        y_file=args.y_file,
    )

    stats = metadata["stats"]
    progress = metadata["progress"]
    p_info = metadata.get("pipeline_info") or {}
    print(
        f"phase4_training {metadata['status']} ({progress['stopped_reason']}): "
        f"epoch {progress['completed_epochs']}/{progress['target_epochs']}, "
        f"loss={stats['latest_loss']}, perplexity={stats['latest_perplexity']}, "
        f"accuracy={stats['latest_accuracy']}, device={stats.get('device', 'CPU')}, "
        f"peak_rss={stats['peak_rss_mb']}MB "
        f"({metadata['elapsed_seconds']}s)"
    )
    if p_info.get("context_file"):
        print(f"context written: {p_info['context_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 新規学習の実行例 (job_id=job_gen3_1234)
PYTHONPATH=impl_gen3/src/pipelines/training/phase4_training \
python -m phase4_training \
  --input-context-file=/tmp/job_gen3_1234/phase3_featurize/context.json \
  --work-directory=/tmp/job_gen3_1234/phase4_training \
  --out-model-file=/tmp/job_gen3_1234/phase4_training/model.keras \
  --out-meta-file=/tmp/job_gen3_1234/phase4_training/metadata.json \
  --out-context-file=/tmp/job_gen3_1234/phase4_training/context.json \
  --input-mode=sparse \
  --epochs=50 \
  --min-delta=0.001 \
  --patience=3 \
  --device=gpu \
  --batch-size=2048

# 中断後の再開実行例 (--resume 指定時は --work-directory のみで再開可能。device と batch-size は変更可能)
PYTHONPATH=impl_gen3/src/pipelines/training/phase4_training \
python -m phase4_training \
  --work-directory=/tmp/job_gen3_1234/phase4_training \
  --resume \
  --device=gpu \
  --batch-size=4096
"""
