"""TensorFlow を起動して GPU が認識・利用可能かを確認するスクリプト。"""

from __future__ import annotations

import tensorflow as tf


def main() -> None:
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Built with CUDA    : {tf.test.is_built_with_cuda()}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Available GPUs ({len(gpus)}): {gpus}")

    if gpus:
        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            device_name = details.get("device_name", "Unknown")
            compute_cap = details.get("compute_capability", "Unknown")
            print(f"  [GPU {i}] {gpu.name} - {device_name} (Compute Capability: {compute_cap})")

        # 実際に GPU 上で簡単なテンソル演算ができるかテスト
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
        print(f"GPU matmul test    : OK (device={c.device})")
    else:
        print("GPU は認識されていません（CPU のみ利用可能）。")


if __name__ == "__main__":
    main()

"""
python sandbox/gpu/check_gpu.py
"""
