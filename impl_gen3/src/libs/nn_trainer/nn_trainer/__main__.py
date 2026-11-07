"""`python -m nn_trainer` で実行するコマンドラインエントリポイント。"""

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
    train_nn,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nn_trainer",
        description=(
            "重み付き One-Hot 特徴量 X (.npz) と正解トークン ID Y (.npy) から "
            "TensorFlow で NN モデルを学習する (1エポックごとに保存・収束判定・中断再開対応)"
        ),
    )
    parser.add_argument(
        "--work-directory",
        required=True,
        help="作業ディレクトリ (model.keras と metadata.json が1エポックごとに保存される)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="中断再開フラグ (指定時は他の引数を無視し、作業ディレクトリの metadata.json から設定を読み込んで再開する)",
    )
    parser.add_argument(
        "--x-file",
        default=None,
        help="入力特徴量 X (.npz) のファイルパス (新規学習時は必須)",
    )
    parser.add_argument(
        "--y-file",
        default=None,
        help="正解ラベル Y (.npy) のファイルパス (新規学習時は必須)",
    )
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
        "--out-model-file",
        default=None,
        help="モデル出力ファイルパス (省略時は {work-directory}/model.keras)",
    )
    parser.add_argument(
        "--out-meta-file",
        default=None,
        help="メタデータ出力ファイルパス (省略時は {work-directory}/metadata.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="ミニバッチサイズ (default: %(default)s)",
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
    args = parser.parse_args(argv)

    if not args.resume:
        if not args.x_file or not args.y_file:
            parser.error(
                "新規学習時 (--resume 未指定時) は --x-file, --y-file の指定が必須です"
            )

    hidden_units = tuple(int(u) for u in args.hidden_units.split(",") if u.strip())

    metadata = train_nn(
        args.work_directory,
        resume=args.resume,
        x_file=args.x_file,
        y_file=args.y_file,
        epochs=args.epochs,
        min_delta=args.min_delta,
        patience=args.patience,
        out_model_file=args.out_model_file,
        out_meta_file=args.out_meta_file,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_units=hidden_units,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        seed=args.seed,
    )

    stats = metadata["stats"]
    progress = metadata["progress"]
    print(
        f"nn_trainer {metadata['status']} ({progress['stopped_reason']}): "
        f"epoch {progress['completed_epochs']}/{progress['target_epochs']}, "
        f"loss={stats['latest_loss']}, perplexity={stats['latest_perplexity']}, "
        f"accuracy={stats['latest_accuracy']}, device={stats.get('device', 'CPU')}, "
        f"peak_rss={stats['peak_rss_mb']}MB "
        f"({metadata['elapsed_seconds']}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
# 新規学習の実行例 (job_id=job_gen3_1234)
PYTHONPATH=impl_gen3/src/libs/nn_trainer \
python -m nn_trainer \
  --work-directory=/tmp/job_gen3_1234/phase4_train \
  --x-file=/tmp/job_gen3_1234/phase3_featurize/x.npz \
  --y-file=/tmp/job_gen3_1234/phase3_featurize/y.npy \
  --epochs=50 \
  --min-delta=0.001 \
  --patience=3

# 中断後の再開実行例 (--resume 指定時は他の引数は不要・指定されても無視して metadata.json から復元)
PYTHONPATH=impl_gen3/src/libs/nn_trainer \
python -m nn_trainer \
  --work-directory=/tmp/job_gen3_1234/phase4_train \
  --resume
"""
