"""TensorFlow (Keras) を用いた SLM ニューラルネットワーク学習モジュール。

Phase 3 (weighted_onehot) で作成した疎行列 X (.npz) と正解トークン ID Y (.npy) を入力とし、
全データでのフルトレーニングを行う。

主な特徴:
- 終了条件として最大エポック数 (epochs) と収束判定閾値 (min_delta, patience) の両方を指定可能
- 1 エポック完了ごとにモデル (.keras) と metadata.json をアトミックに保存する
- 中断・再開フラグ (resume=True / --resume) が指定された場合は、他の引数を無視して
  作業ディレクトリ内の metadata.json から設定・進捗・保存済みモデル（オプティマイザ状態含む）を
  読み込んで続きのエポックから学習を再開する
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

# libs 配下の共通統計モジュール (runtime_stats) を単体実行時でも import できるようにパスを補完する
_LIBS_DIR = Path(__file__).resolve().parents[2]
for _lib_name in ("runtime_stats",):
    _lib_path = _LIBS_DIR / _lib_name
    if _lib_path.is_dir() and str(_lib_path) not in sys.path:
        sys.path.insert(0, str(_lib_path))

from runtime_stats import get_memory_stats  # noqa: E402

__all__ = [
    "DEFAULT_EPOCHS",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_PATIENCE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_HIDDEN_UNITS",
    "DEFAULT_ACTIVATION",
    "DEFAULT_DROPOUT_RATE",
    "DEFAULT_STEPS_PER_EXECUTION",
    "CSRBatchSequence",
    "build_csr_tf_dataset",
    "configure_gpu_memory_growth",
    "build_model",
    "train_nn",
]

DEFAULT_EPOCHS = 50
DEFAULT_MIN_DELTA = 1e-3
DEFAULT_PATIENCE = 3
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_UNITS = (256, 256)
DEFAULT_ACTIVATION = "gelu"
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_STEPS_PER_EXECUTION = 64


def configure_gpu_memory_growth() -> list[str]:
    """利用可能な GPU に対してメモリ動的確保 (memory_growth) を設定し、GPU デバイス名一覧を返す。"""
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names: list[str] = []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # すでにデバイス初期化済みの場合は設定変更できないため無視して続行
            pass
        gpu_names.append(getattr(gpu, "name", str(gpu)))
    return gpu_names


class CSRBatchSequence(tf.keras.utils.Sequence):
    """CSR 疎行列 X と 1次元配列 Y からミニバッチ単位で密配列に変換して供給するデータローダ。

    全データを一括で密行列化しないため省メモリで動作し、Keras 標準の Dense 層や
    モデル保存・読み込み (load_model) と完全に互換性を持つ。
    また、`.toarray()` の累積時間 (`toarray_seconds`) と `on_epoch_end()` の
    CSR 全件並べ替え時間 (`reorder_seconds`) を計測・保持する。
    """

    def __init__(
        self,
        x_csr: sp.csr_matrix,
        y_arr: np.ndarray,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.x_csr = x_csr
        self.y_arr = np.asarray(y_arr, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_samples = int(self.x_csr.shape[0])
        self._rng = np.random.default_rng(seed)
        self.indices = np.arange(self.num_samples, dtype=np.int32)
        self.toarray_seconds: float = 0.0
        self.reorder_seconds: float = 0.0
        if self.shuffle:
            t0 = time.perf_counter()
            self._rng.shuffle(self.indices)
            self._x_epoch = self.x_csr[self.indices]
            self._y_epoch = self.y_arr[self.indices]
            self.reorder_seconds += time.perf_counter() - t0
        else:
            self._x_epoch = self.x_csr
            self._y_epoch = self.y_arr

    def __len__(self) -> int:
        if self.num_samples == 0:
            return 0
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        end = min(start + self.batch_size, self.num_samples)
        t0 = time.perf_counter()
        x_batch = self._x_epoch[start:end].toarray().astype(np.float32, copy=False)
        self.toarray_seconds += time.perf_counter() - t0
        y_batch = self._y_epoch[start:end]
        return x_batch, y_batch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            t0 = time.perf_counter()
            self._rng.shuffle(self.indices)
            self._x_epoch = self.x_csr[self.indices]
            self._y_epoch = self.y_arr[self.indices]
            self.reorder_seconds += time.perf_counter() - t0


def build_csr_tf_dataset(
    indices: np.ndarray,
    *,
    batch_size: int,
    max_row_nnz: int,
    csr_indptr_t: tf.Tensor,
    csr_indices_t: tf.Tensor,
    csr_data_t: tf.Tensor,
    y_t: tf.Tensor,
) -> tf.data.Dataset:
    """CSR 1次元配列群から固定幅 (batch_size, max_row_nnz) の疎バッチを高速抽出する tf.data.Dataset を構築する。

    CPU 側での `.toarray()` による巨大な密行列生成と PyFunc オーバーヘッド、および
    PCIe バス上の密行列転送 (約282GB/epoch) を回避し、非ゼロ要素の列番号と重みのみ
    (1バッチあたり約64KB) を GPU へ転送する。
    """
    k_range = tf.expand_dims(tf.range(max_row_nnz, dtype=tf.int32), axis=0)  # [1, K]
    row_ds = tf.data.Dataset.from_tensor_slices(indices)
    batch_row_ds = row_ds.batch(batch_size, drop_remainder=False)

    def _slice_sparse_2d(batch_rows: tf.Tensor):
        starts = tf.expand_dims(tf.gather(csr_indptr_t, batch_rows), axis=1)  # [B, 1]
        ends = tf.expand_dims(tf.gather(csr_indptr_t, batch_rows + 1), axis=1)  # [B, 1]
        valid = k_range < (ends - starts)  # [B, K]
        safe_pos = tf.where(valid, starts + k_range, 0)  # [B, K]
        col_padded = tf.where(valid, tf.gather(csr_indices_t, safe_pos), 0)  # [B, K]
        val_padded = tf.where(valid, tf.gather(csr_data_t, safe_pos), 0.0)  # [B, K]
        y_batch = tf.gather(y_t, batch_rows)
        return (col_padded, val_padded), y_batch

    return batch_row_ds.map(
        _slice_sparse_2d,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def _attach_gpu_sparse_train_step(
    model: tf.keras.Model,
    *,
    vocab_size: int,
    max_row_nnz: int,
    steps_per_execution: int = DEFAULT_STEPS_PER_EXECUTION,
) -> None:
    """モデルのネットワーク構造や保存形式を変えずに、GPU (XLA) 上で疎バッチを密テンソルへ展開してから train_step を実行するラッパを設定する。"""
    orig_train_step = model.train_step

    def _densify_batch(col_padded: tf.Tensor, val_padded: tf.Tensor) -> tf.Tensor:
        actual_bs = tf.shape(col_padded)[0]
        row_ids = tf.broadcast_to(
            tf.expand_dims(tf.range(actual_bs, dtype=tf.int32), axis=1),
            [actual_bs, max_row_nnz],
        )
        coords = tf.stack([row_ids, col_padded], axis=2)  # [B, K, 2]
        zeros = tf.zeros([actual_bs, vocab_size], dtype=tf.float32)
        return tf.tensor_scatter_nd_add(zeros, coords, val_padded)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _compiled_sparse_step(col_padded: tf.Tensor, val_padded: tf.Tensor, y_in: tf.Tensor):
        x_dense = _densify_batch(col_padded, val_padded)
        return orig_train_step((x_dense, y_in))

    def _sparse_aware_train_step(data):
        x_in, y_in = data
        if isinstance(x_in, (tuple, list)) and len(x_in) == 2:
            return _compiled_sparse_step(x_in[0], x_in[1], y_in)
        return orig_train_step((x_in, y_in))

    model.train_step = _sparse_aware_train_step
    model.jit_compile = False
    model.steps_per_execution = int(steps_per_execution)
    model.train_function = None


def build_model(
    vocab_size: int,
    *,
    hidden_units: tuple[int, ...] | list[int] = DEFAULT_HIDDEN_UNITS,
    activation: str = DEFAULT_ACTIVATION,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """重み付き One-Hot ベクトル (vocab_size 次元) から次単語を予測する NN モデルを構築する。

    構造:
        Input(vocab_size)
        -> [Dense(u, activation) -> LayerNormalization -> Dropout] x N
        -> Dense(vocab_size, activation="softmax")
    """
    inputs = tf.keras.Input(shape=(vocab_size,), dtype=tf.float32, name="weighted_onehot_input")
    h = inputs
    for idx, units in enumerate(hidden_units, start=1):
        h = tf.keras.layers.Dense(
            int(units),
            activation=activation,
            name=f"hidden_dense_{idx}",
        )(h)
        h = tf.keras.layers.LayerNormalization(name=f"hidden_ln_{idx}")(h)
        if dropout_rate > 0.0:
            h = tf.keras.layers.Dropout(float(dropout_rate), name=f"hidden_dropout_{idx}")(h)

    outputs = tf.keras.layers.Dense(
        vocab_size,
        activation="softmax",
        name="token_prob_output",
    )(h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="slm_gen1_mlp")
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(learning_rate))
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def _atomic_save_json(data: dict, target_path: Path) -> None:
    """中断時に JSON が破損しないよう一時ファイル経由でアトミックに書き出す。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(target_path)


def _atomic_save_model(model: tf.keras.Model, target_path: Path) -> None:
    """中断時にモデルファイルが破損しないよう一時ファイル経由でアトミックに保存する。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(target_path.stem + ".tmp" + target_path.suffix)
    model.save(str(tmp_path))
    if tmp_path.is_dir():
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.move(str(tmp_path), str(target_path))
    else:
        tmp_path.replace(target_path)


def _recompute_convergence_state(
    history_records: list[dict],
    min_delta: float,
) -> tuple[float | None, int]:
    """履歴から (best_loss_anchor, no_improve_count) を再計算する。"""
    best_anchor: float | None = None
    no_improve_count = 0
    for rec in history_records:
        loss = float(rec["loss"])
        if best_anchor is None or (best_anchor - loss) >= min_delta:
            best_anchor = loss if best_anchor is None else min(best_anchor, loss)
            no_improve_count = 0
        else:
            best_anchor = min(best_anchor, loss)
            no_improve_count += 1
    return best_anchor, no_improve_count


def train_nn(
    work_directory: str | Path,
    *,
    resume: bool = False,
    x_file: str | Path | None = None,
    y_file: str | Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    min_delta: float = DEFAULT_MIN_DELTA,
    patience: int = DEFAULT_PATIENCE,
    out_model_file: str | Path | None = None,
    out_meta_file: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    hidden_units: tuple[int, ...] | list[int] = DEFAULT_HIDDEN_UNITS,
    activation: str = DEFAULT_ACTIVATION,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    seed: int = 42,
    pipeline_info: dict | None = None,
) -> dict:
    """NN の学習を実行し、1 エポックごとにモデルと metadata.json を保存する。

    終了条件:
        1. completed_epochs が epochs (最大エポック数) に達した場合
        2. min_delta > 0 かつ Loss の改善幅が min_delta 未満のエポックが patience 回連続した場合

    Args:
        work_directory: 作業ディレクトリ (model.keras と metadata.json が保存される)
        resume: True の場合、他の引数をすべて無視して work_directory 内の metadata.json から設定を読み込み学習を再開する
        x_file: 入力特徴量 X (.npz) のファイルパス (resume=False 時は必須)
        y_file: 正解ラベル Y (.npy) のファイルパス (resume=False 時は必須)
        epochs: 最大エポック数 (default: 50)
        min_delta: 収束判定の最小 Loss 改善幅 (0 以下で無効化、default: 0.001)
        patience: 改善幅 < min_delta が何エポック連続したら収束終了とするか (default: 3)
        out_model_file: モデル出力ファイルパス (省略時は {work_directory}/model.keras)
        out_meta_file: メタデータ出力ファイルパス (省略時は {work_directory}/metadata.json)
        batch_size: ミニバッチサイズ
        learning_rate: 学習率
        hidden_units: 隠れ層のユニット数タプル
        activation: 隠れ層の活性化関数
        dropout_rate: ドロップアウト率
        seed: 乱数シード

    Returns:
        最新のメタデータ辞書
    """
    work_dir = Path(work_directory)
    default_meta_path = work_dir / "metadata.json"

    if resume:
        # 再開モード: 他の引数は無視し、保存されている metadata.json から全設定を復元する
        meta_path = Path(out_meta_file) if out_meta_file is not None else default_meta_path
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"再開用の metadata.json が見つかりません: {meta_path}"
            )
        with meta_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

        resolved_x_file = Path(metadata["input"]["x_file"])
        resolved_y_file = Path(metadata["input"]["y_file"])
        resolved_model_file = Path(metadata["outputs"]["model_file"])
        resolved_meta_file = Path(metadata["outputs"]["meta_file"])

        params = metadata["params"]
        target_epochs = int(params["epochs"])
        min_delta = float(params.get("min_delta", DEFAULT_MIN_DELTA))
        patience = int(params.get("patience", DEFAULT_PATIENCE))
        batch_size = int(params["batch_size"])
        learning_rate = float(params["learning_rate"])
        hidden_units = tuple(int(u) for u in params["hidden_units"])
        activation = str(params["activation"])
        dropout_rate = float(params["dropout_rate"])
        seed = int(params.get("seed", 42))

        completed_epochs = int(metadata["progress"]["completed_epochs"])
        stopped_reason = metadata["progress"].get("stopped_reason")
        history_records: list[dict] = list(metadata.get("history", []))
        cumulative_elapsed = float(metadata.get("elapsed_seconds", 0.0))
        started_at_str = str(metadata["started_at"])
        resume_count = int(metadata.get("resume_count", 0)) + 1
        prev_memory_stats: dict | None = (
            metadata.get("stats", {}).get("memory") or metadata.get("stats")
        )
        saved_pipeline_info = metadata.get("pipeline_info")
        if saved_pipeline_info and pipeline_info is None:
            pipeline_info = dict(saved_pipeline_info)

        if not resolved_model_file.exists():
            raise FileNotFoundError(
                f"再開用の保存済みモデルが見つかりません: {resolved_model_file}"
            )
    else:
        # 新規学習モード
        if x_file is None or y_file is None:
            raise ValueError("新規学習時 (resume=False) は x_file と y_file の指定が必須です")
        if epochs is None or int(epochs) < 1:
            raise ValueError(f"新規学習時 (resume=False) は epochs >= 1 の指定が必須です: {epochs}")
        if int(patience) < 1:
            raise ValueError(f"patience は 1 以上を指定してください: {patience}")

        resolved_x_file = Path(x_file)
        resolved_y_file = Path(y_file)
        resolved_model_file = (
            Path(out_model_file) if out_model_file is not None else work_dir / "model.keras"
        )
        resolved_meta_file = (
            Path(out_meta_file) if out_meta_file is not None else default_meta_path
        )

        target_epochs = int(epochs)
        min_delta = float(min_delta)
        patience = int(patience)
        batch_size = int(batch_size)
        learning_rate = int(learning_rate) if isinstance(learning_rate, bool) else float(learning_rate)
        hidden_units = tuple(int(u) for u in hidden_units)
        activation = str(activation)
        dropout_rate = float(dropout_rate)
        seed = int(seed)

        completed_epochs = 0
        stopped_reason = None
        history_records = []
        cumulative_elapsed = 0.0
        started_at_str = datetime.now(timezone.utc).isoformat()
        resume_count = 0
        prev_memory_stats = None

    best_anchor, no_improve_count = _recompute_convergence_state(history_records, min_delta)

    # データの読み込み時間計測
    t_load_start = time.perf_counter()
    x_csr = sp.load_npz(str(resolved_x_file)).tocsr()
    x_csr.sum_duplicates()
    y_arr = np.asarray(np.load(str(resolved_y_file)), dtype=np.int32)
    data_load_seconds = time.perf_counter() - t_load_start

    if x_csr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X と Y のサンプル数が一致しません: X={x_csr.shape}, Y={y_arr.shape}"
        )

    num_samples = int(x_csr.shape[0])
    vocab_size = int(x_csr.shape[1])
    num_batches = math.ceil(num_samples / batch_size) if num_samples > 0 else 0
    row_lengths = np.diff(x_csr.indptr)
    max_row_nnz = max(1, int(row_lengths.max())) if row_lengths.size > 0 else 1

    gpu_devices = configure_gpu_memory_growth()
    device_type = "GPU" if gpu_devices else "CPU"

    # CSR 1次元配列群の Tensor 化時間計測 (全件密行列化は行わず約1.1GBの疎配列のまま保持)
    t_csr_init_start = time.perf_counter()
    with tf.device("/CPU:0"):
        csr_indptr_t = tf.constant(x_csr.indptr, dtype=tf.int32)
        csr_indices_t = tf.constant(x_csr.indices, dtype=tf.int32)
        csr_data_t = tf.constant(x_csr.data, dtype=tf.float32)
        y_t = tf.constant(y_arr, dtype=tf.int32)
    csr_tensor_init_seconds = time.perf_counter() - t_csr_init_start

    tf.keras.utils.set_random_seed(seed + completed_epochs)

    # モデルの構築またはロード時間計測
    t_model_init_start = time.perf_counter()
    if resume:
        model = tf.keras.models.load_model(str(resolved_model_file))
    else:
        model = build_model(
            vocab_size,
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
    _attach_gpu_sparse_train_step(
        model,
        vocab_size=vocab_size,
        max_row_nnz=max_row_nnz,
        steps_per_execution=DEFAULT_STEPS_PER_EXECUTION,
    )
    model_init_seconds = time.perf_counter() - t_model_init_start

    rng = np.random.default_rng(seed + completed_epochs)
    indices = np.arange(num_samples, dtype=np.int32)

    def _build_timing_summary() -> dict:
        total_shuffle_sec = sum(float(r.get("shuffle_and_dataset_seconds", 0.0)) for r in history_records)
        total_fit_sec = sum(float(r.get("train_fit_seconds", 0.0)) for r in history_records)
        total_save_sec = sum(float(r.get("model_save_seconds", 0.0)) for r in history_records)
        latest_rec = history_records[-1] if history_records else {}
        return {
            "data_load_seconds": round(data_load_seconds, 3),
            "csr_tensor_init_seconds": round(csr_tensor_init_seconds, 3),
            "model_init_seconds": round(model_init_seconds, 3),
            "total_shuffle_and_dataset_seconds": round(total_shuffle_sec, 3),
            "total_train_fit_seconds": round(total_fit_sec, 3),
            "total_model_save_seconds": round(total_save_sec, 3),
            "latest_epoch_shuffle_and_dataset_seconds": latest_rec.get("shuffle_and_dataset_seconds"),
            "latest_epoch_train_fit_seconds": latest_rec.get("train_fit_seconds"),
            "latest_epoch_ms_per_step": latest_rec.get("ms_per_step"),
            "latest_epoch_model_save_seconds": latest_rec.get("model_save_seconds"),
        }

    def _build_metadata_dict(status: str, reason: str | None) -> dict:
        nonlocal prev_memory_stats
        now_iso = datetime.now(timezone.utc).isoformat()
        mem_stats = get_memory_stats(prev_memory_stats)
        prev_memory_stats = mem_stats
        timing_summary = _build_timing_summary()
        if history_records:
            latest = history_records[-1]
            best = min(history_records, key=lambda r: r["loss"])
            stats_summary = {
                "num_samples": num_samples,
                "vocab_size": vocab_size,
                "num_batches_per_epoch": num_batches,
                "max_row_nnz": max_row_nnz,
                "device": device_type,
                "gpu_devices": gpu_devices,
                "latest_epoch": latest["epoch"],
                "latest_loss": latest["loss"],
                "latest_perplexity": latest["perplexity"],
                "latest_accuracy": latest["accuracy"],
                "best_epoch": best["epoch"],
                "best_loss": best["loss"],
                "best_perplexity": best["perplexity"],
                "best_accuracy": best["accuracy"],
                "peak_rss_kb": mem_stats["peak_rss_kb"],
                "peak_rss_mb": mem_stats["peak_rss_mb"],
                "current_rss_kb": mem_stats["current_rss_kb"],
                "current_rss_mb": mem_stats["current_rss_mb"],
                "peak_vm_kb": mem_stats["peak_vm_kb"],
                "peak_vm_mb": mem_stats["peak_vm_mb"],
                "memory": mem_stats,
                "timing": timing_summary,
            }
        else:
            stats_summary = {
                "num_samples": num_samples,
                "vocab_size": vocab_size,
                "num_batches_per_epoch": num_batches,
                "max_row_nnz": max_row_nnz,
                "device": device_type,
                "gpu_devices": gpu_devices,
                "latest_epoch": 0,
                "latest_loss": None,
                "latest_perplexity": None,
                "latest_accuracy": None,
                "best_epoch": None,
                "best_loss": None,
                "best_perplexity": None,
                "best_accuracy": None,
                "peak_rss_kb": mem_stats["peak_rss_kb"],
                "peak_rss_mb": mem_stats["peak_rss_mb"],
                "current_rss_kb": mem_stats["current_rss_kb"],
                "current_rss_mb": mem_stats["current_rss_mb"],
                "peak_vm_kb": mem_stats["peak_vm_kb"],
                "peak_vm_mb": mem_stats["peak_vm_mb"],
                "memory": mem_stats,
                "timing": timing_summary,
            }

        return {
            "status": status,
            "started_at": started_at_str,
            "updated_at": now_iso,
            "finished_at": now_iso if status == "completed" else None,
            "elapsed_seconds": round(cumulative_elapsed, 3),
            "resume_count": resume_count,
            "pipeline_info": pipeline_info,
            "input": {
                "x_file": str(resolved_x_file),
                "y_file": str(resolved_y_file),
            },
            "outputs": {
                "work_directory": str(work_dir),
                "model_file": str(resolved_model_file),
                "meta_file": str(resolved_meta_file),
            },
            "params": {
                "epochs": target_epochs,
                "min_delta": min_delta,
                "patience": patience,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_units": list(hidden_units),
                "activation": activation,
                "dropout_rate": dropout_rate,
                "seed": seed,
            },
            "progress": {
                "completed_epochs": completed_epochs,
                "target_epochs": target_epochs,
                "remaining_epochs": max(0, target_epochs - completed_epochs),
                "no_improve_count": no_improve_count,
                "stopped_reason": reason,
            },
            "stats": stats_summary,
            "history": history_records,
        }

    def _save_metadata_files(meta_dict: dict) -> None:
        _atomic_save_json(meta_dict, resolved_meta_file)
        if resolved_meta_file.resolve() != default_meta_path.resolve():
            _atomic_save_json(meta_dict, default_meta_path)

    # すでに目標エポック数または収束条件に達している場合はそのまま完了状態で返す
    if completed_epochs >= target_epochs:
        final_meta = _build_metadata_dict("completed", stopped_reason or "reached_max_epochs")
        _save_metadata_files(final_meta)
        return final_meta
    if min_delta > 0.0 and no_improve_count >= patience:
        final_meta = _build_metadata_dict(
            "completed",
            stopped_reason or f"converged_min_delta (min_delta={min_delta}, patience={patience})",
        )
        _save_metadata_files(final_meta)
        return final_meta

    # 初期状態の metadata.json を保存
    _save_metadata_files(_build_metadata_dict("running", None))

    # 1 エポックずつ学習 → 収束判定 → モデル保存 → metadata.json 更新
    for current_epoch in range(completed_epochs + 1, target_epochs + 1):
        epoch_start = time.perf_counter()

        # インデックスのシャッフルと tf.data.Dataset の構築時間計測
        t_shuf_start = time.perf_counter()
        rng.shuffle(indices)
        epoch_dataset = build_csr_tf_dataset(
            indices,
            batch_size=batch_size,
            max_row_nnz=max_row_nnz,
            csr_indptr_t=csr_indptr_t,
            csr_indices_t=csr_indices_t,
            csr_data_t=csr_data_t,
            y_t=y_t,
        )
        shuffle_and_dataset_sec = time.perf_counter() - t_shuf_start

        # NN 学習 (model.fit) 時間計測
        t_fit_start = time.perf_counter()
        hist = model.fit(
            epoch_dataset,
            epochs=current_epoch,
            initial_epoch=current_epoch - 1,
            shuffle=False,
            verbose=1,
        )
        train_fit_sec = time.perf_counter() - t_fit_start
        ms_per_step = (train_fit_sec / num_batches * 1000.0) if num_batches > 0 else 0.0

        loss_val = float(hist.history["loss"][-1])
        acc_val = float(hist.history["accuracy"][-1])
        ppl_val = float(math.exp(min(loss_val, 50.0)))

        if best_anchor is None or (best_anchor - loss_val) >= min_delta:
            loss_improvement = None if best_anchor is None else round(best_anchor - loss_val, 6)
            best_anchor = loss_val if best_anchor is None else min(best_anchor, loss_val)
            no_improve_count = 0
        else:
            loss_improvement = round(best_anchor - loss_val, 6)
            best_anchor = min(best_anchor, loss_val)
            no_improve_count += 1

        # 終了条件の判定 (収束 or 最大エポック到達)
        completed_epochs = current_epoch
        converged = min_delta > 0.0 and no_improve_count >= patience
        reached_max = completed_epochs >= target_epochs

        if converged:
            stopped_reason = f"converged_min_delta (min_delta={min_delta}, patience={patience})"
            status = "completed"
        elif reached_max:
            stopped_reason = "reached_max_epochs"
            status = "completed"
        else:
            stopped_reason = None
            status = "running"

        # モデル保存時間計測
        t_save_start = time.perf_counter()
        _atomic_save_model(model, resolved_model_file)
        model_save_sec = time.perf_counter() - t_save_start

        epoch_elapsed = time.perf_counter() - epoch_start
        cumulative_elapsed += epoch_elapsed

        epoch_mem = get_memory_stats(prev_memory_stats)
        prev_memory_stats = epoch_mem
        history_records.append(
            {
                "epoch": current_epoch,
                "loss": round(loss_val, 6),
                "loss_improvement": loss_improvement,
                "no_improve_count": no_improve_count,
                "perplexity": round(ppl_val, 4),
                "accuracy": round(acc_val, 6),
                "peak_rss_mb": epoch_mem["peak_rss_mb"],
                "current_rss_mb": epoch_mem["current_rss_mb"],
                "shuffle_and_dataset_seconds": round(shuffle_and_dataset_sec, 3),
                "train_fit_seconds": round(train_fit_sec, 3),
                "ms_per_step": round(ms_per_step, 3),
                "model_save_seconds": round(model_save_sec, 3),
                "elapsed_seconds": round(epoch_elapsed, 3),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # metadata.json を保存
        current_meta = _build_metadata_dict(status, stopped_reason)
        _save_metadata_files(current_meta)

        if status == "completed":
            return current_meta

    return _build_metadata_dict("completed", stopped_reason or "reached_max_epochs")
