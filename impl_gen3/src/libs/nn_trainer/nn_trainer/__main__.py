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
            "TensorFlow で NN モデルを学習する (チャンクごとに保存・再開、全チャンク完了後に収束判定)"
        ),
    )
    parser.add_argument(
        "--work-directory",
        required=True,
        help="作業ディレクトリ (model.keras と metadata.json がチャンクごとに保存される)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="中断再開フラグ (指定時は device、batch-size と出力メタデータ指定以外の引数を無視し、作業ディレクトリの metadata.json から設定を読み込んで再開する)",
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
    parser.add_argument("--x-files", nargs="+", help="順序付き X チャンク一覧")
    parser.add_argument("--y-files", nargs="+", help="順序付き Y チャンク一覧")
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
        if not ((args.x_file and args.y_file) or (args.x_files and args.y_files)):
            parser.error(
                "新規学習時 (--resume 未指定時) は --x-file, --y-file の指定が必須です"
            )

    hidden_units = tuple(int(u) for u in args.hidden_units.split(",") if u.strip())

    metadata = train_nn(
        args.work_directory,
        resume=args.resume,
        x_file=args.x_file,
        y_file=args.y_file,
        x_files=args.x_files,
        y_files=args.y_files,
        epochs=args.epochs,
        min_delta=args.min_delta,
        patience=args.patience,
        out_model_file=args.out_model_file,
        out_meta_file=args.out_meta_file,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_units=hidden_units,
        input_mode=args.input_mode,
        device=args.device,
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
  --input-mode=sparse \
  --device=auto \
  --epochs=50 \
  --min-delta=0.001 \
  --patience=3 \
  --batch-size=2048

# 中断後の再開実行例 (--resume 指定時はdevice、batch-size と出力メタデータ指定以外の引数は metadata.json から復元)
PYTHONPATH=impl_gen3/src/libs/nn_trainer \
python -m nn_trainer \
  --work-directory=/tmp/job_gen3_1234/phase4_train \
  --resume
"""
