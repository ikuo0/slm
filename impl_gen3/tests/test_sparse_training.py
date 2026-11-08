"""先頭層の疎計算について、密計算との勾配一致と再開・推論互換性を検証する。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC / "libs" / "nn_trainer"))
sys.path.insert(0, str(SRC / "pipelines" / "training" / "phase4_training"))
import nn_trainer as trainer
from phase4_training import run_phase4_training


class SparseTrainingTests(unittest.TestCase):
    def test_matches_dense_loss_metrics_and_weight_update(self):
        # 重複インデックス、負の重み、padding、全ゼロ行を含む。
        columns = tf.constant([[1, 1, 4], [2, 0, 0], [0, 0, 0]])
        values = tf.constant([[0.2, 0.3, -0.4], [0.8, 0., 0.], [0., 0., 0.]])
        dense = np.zeros((3, 7), dtype=np.float32)
        for row, (cols, vals) in enumerate(zip(columns.numpy(), values.numpy())):
            np.add.at(dense[row], cols, vals)
        labels = tf.constant([1, 3, 0])
        for hidden in ((5, 3), ()):
            with self.subTest(hidden=hidden):
                a = trainer.build_model(7, hidden_units=hidden, dropout_rate=0)
                b = trainer.build_model(7, hidden_units=hidden, dropout_rate=0)
                b.set_weights(a.get_weights())
                # SGD により、optimizer の疎更新実装から独立に勾配一致を比較する。
                for model in (a, b):
                    model.compile(optimizer=tf.keras.optimizers.SGD(0.02),
                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
                trainer._attach_direct_sparse_train_step(b)
                expected = a.train_on_batch(dense, labels, return_dict=True)
                actual = b.train_on_batch((columns, values), labels, return_dict=True)
                for key in expected:
                    np.testing.assert_allclose(actual[key], expected[key], atol=2e-6)
                for aw, bw in zip(a.get_weights(), b.get_weights()):
                    np.testing.assert_allclose(aw, bw, atol=2e-6, rtol=2e-5)

    def test_phase4_sparse_checkpoint_resume_and_dense_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.array([[0., 0.5, 0., 0.2], [0., 0., 1., 0.], [0., 0., 0., 0.]], dtype=np.float32)
            xp, yp = root / "x.npz", root / "y.npy"
            sp.save_npz(xp, sp.csr_matrix(x))
            np.save(yp, np.array([1, 2, 0], dtype=np.int32))
            original_save = trainer._atomic_save_json

            def interrupt_after_chunk(data, path):
                original_save(data, path)
                if data["progress"]["next_chunk"] == 1:
                    raise KeyboardInterrupt

            with patch.object(trainer, "_atomic_save_json", side_effect=interrupt_after_chunk):
                with self.assertRaises(KeyboardInterrupt):
                    run_phase4_training(root / "work", x_file=xp, y_file=yp,
                                        input_mode="sparse", epochs=2, min_delta=0,
                                        batch_size=2, hidden_units=(3,), dropout_rate=0)
            saved = json.loads((root / "work" / "metadata.json").read_text())
            self.assertEqual(saved["params"]["input_mode"], "sparse")
            # デフォルト dense を渡しても、再開時は保存された sparse が使われる。
            with patch.object(trainer, "_attach_gpu_sparse_train_step", side_effect=AssertionError("dense path")):
                result = run_phase4_training(root / "work", resume=True, device="cpu", batch_size=3)
            self.assertEqual(result["params"]["batch_size"], 3)
            self.assertEqual(result["params"]["device"], "cpu")
            self.assertEqual(result["stats"]["device"], "CPU")
            self.assertEqual(result["stats"]["gpu_devices"], [])
            self.assertEqual(result["params"]["input_mode"], "sparse")
            self.assertEqual(result["progress"]["completed_epochs"], 2)
            model = tf.keras.models.load_model(result["outputs"]["model_file"])
            self.assertEqual(int(model.optimizer.iterations.numpy()), 3)
            predictions = model(x, training=False).numpy()
            self.assertEqual(predictions.shape, (3, 4))
            np.testing.assert_allclose(predictions.sum(axis=1), 1., atol=1e-6)


if __name__ == "__main__":
    unittest.main()
