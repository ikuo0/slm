"""実行: python -m unittest discover -s impl_gen3/tests -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

SRC = Path(__file__).resolve().parents[1] / "src"
for path in (SRC / "libs").iterdir():
    if path.is_dir():
        sys.path.insert(0, str(path))
for phase in ("phase3_featurize", "phase4_training"):
    sys.path.insert(0, str(SRC / "pipelines" / "training" / phase))
import nn_trainer as trainer
import weighted_onehot as features
from phase3_featurize import run_phase3_featurize
from phase4_training import resolve_phase3_xy_files, run_phase4_training


class ChunkTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.xs, self.ys = [], []
        for i, size in enumerate((3, 1), 1):
            xp, yp = self.root / f"x_{i:04}.npz", self.root / f"y_{i:04}.npy"
            sp.save_npz(xp, sp.csr_matrix(np.ones((size, 4), dtype=np.float32)))
            np.save(yp, np.zeros(size, dtype=np.int32))
            self.xs.append(str(xp))
            self.ys.append(str(yp))

    def test_resume_and_weighted_epoch_loss(self):
        calls = []
        interrupt = [True]
        class Model:
            def __init__(self, trained=0):
                self.trained = trained
            def save(self, path):
                Path(path).write_text(str(self.trained))
            def fit(self, dataset, **kwargs):
                if self.trained == 1 and interrupt[0]:
                    raise KeyboardInterrupt
                calls.append(self.trained)
                loss = (2.0, 6.0)[self.trained % 2]
                self.trained += 1
                return SimpleNamespace(history={"loss": [loss], "accuracy": [0.5]})
        work = self.root / "work"
        context = self.root / "context.json"
        context.write_text(json.dumps({"phase3_featurize": {"x_files": self.xs, "y_files": self.ys,
                                                                        "x_file": self.xs[0], "y_file": self.ys[0]}}))
        with patch.object(trainer, "build_model", side_effect=lambda *a, **k: Model()), \
             patch.object(trainer, "_attach_gpu_sparse_train_step"), \
             patch.object(trainer.tf.keras.models, "load_model", side_effect=lambda p: Model(int(Path(p).read_text()))):
            with self.assertRaises(KeyboardInterrupt):
                run_phase4_training(work, input_context_file=context, epochs=3, patience=1)
            saved = json.loads((work / "metadata.json").read_text())
            self.assertEqual(saved["progress"]["next_chunk"], 1)
            self.assertEqual(saved["progress"]["completed_epochs"], 0)
            self.assertEqual(saved["history"], [])
            self.assertEqual(saved["progress"]["no_improve_count"], 0)
            interrupt[0] = False
            result = run_phase4_training(work, resume=True)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(result["progress"]["completed_epochs"], 2)
        self.assertEqual([r["loss"] for r in result["history"]], [3.0, 3.0])
        self.assertEqual(result["progress"]["next_chunk"], 0)
        self.assertEqual(result["progress"]["partial_epoch"], {})
        self.assertEqual((work / "model.keras").read_text(), "4")

    def test_resume_batch_override_and_omission(self):
        for override in (None, 2):
            with self.subTest(override=override):
                batches = []
                interrupt = [True]

                class Model:
                    def __init__(self, trained=0):
                        self.trained = trained

                    def save(self, path):
                        Path(path).write_text(str(self.trained))

                    def fit(self, dataset, **kwargs):
                        if self.trained == 1 and interrupt[0]:
                            raise KeyboardInterrupt
                        batches.append([int(y.shape[0]) for _, y in dataset])
                        self.trained += 1
                        return SimpleNamespace(history={"loss": [2.0], "accuracy": [0.5]})

                work = self.root / f"batch_{override}"
                context = self.root / f"batch_context_{override}.json"
                context.write_text(json.dumps({"phase3_featurize": {"x_files": self.xs, "y_files": self.ys}}))
                with patch.object(trainer, "build_model", return_value=Model()), \
                     patch.object(trainer, "_attach_gpu_sparse_train_step"), \
                     patch.object(trainer.tf.keras.models, "load_model",
                                  side_effect=lambda path: Model(int(Path(path).read_text()))):
                    with self.assertRaises(KeyboardInterrupt):
                        run_phase4_training(work, input_context_file=context, epochs=2, min_delta=0,
                                            batch_size=1)
                    # 1つ目のチャンク完了時点から、途中エポックを再開。
                    saved = json.loads((work / "metadata.json").read_text())
                    self.assertEqual(saved["params"]["batch_size"], 1)
                    # 旧メタデータ形式も扱えることを検証。
                    saved["progress"]["partial_epoch"].pop("num_batches", None)
                    (work / "metadata.json").write_text(json.dumps(saved))
                    with self.assertRaises(ValueError):
                        run_phase4_training(work, resume=True, batch_size=0)
                    interrupt[0] = False
                    result = run_phase4_training(work, resume=True, batch_size=override)
                expected_size = 1 if override is None else override
                self.assertEqual(result["params"]["batch_size"], expected_size)
                self.assertEqual(batches, [[1, 1, 1], [1],
                                          [1, 1, 1] if override is None else [2, 1], [1]])
                self.assertEqual([r["num_batches"] for r in result["history"]],
                                 [4, 4 if override is None else 3])
                self.assertEqual(result["progress"]["completed_epochs"], 2)
                self.assertEqual(result["history"][0]["loss"], 2.0)
                with patch.object(trainer, "_attach_gpu_sparse_train_step"), \
                     patch.object(trainer.tf.keras.models, "load_model", return_value=Model()):
                    again = run_phase4_training(work, resume=True)
                self.assertEqual(again["params"]["batch_size"], expected_size)

    def test_interrupt_after_final_chunk_before_epoch_judgment(self):
        calls = []
        class Model:
            def save(self, path):
                Path(path).write_text("saved")
            def fit(self, dataset, **kwargs):
                calls.append(1)
                return SimpleNamespace(history={"loss": [2.0], "accuracy": [0.5]})
        original_save = trainer._atomic_save_json
        def save_then_interrupt(data, path):
            original_save(data, path)
            if data["progress"]["next_chunk"] == 2:
                raise KeyboardInterrupt
        work = self.root / "boundary"
        with patch.object(trainer, "build_model", return_value=Model()), \
             patch.object(trainer, "_attach_gpu_sparse_train_step"), \
             patch.object(trainer.tf.keras.models, "load_model", return_value=Model()):
            with patch.object(trainer, "_atomic_save_json", side_effect=save_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    trainer.train_nn(work, x_files=self.xs, y_files=self.ys, epochs=1)
            saved = json.loads((work / "metadata.json").read_text())
            self.assertEqual(saved["progress"]["completed_epochs"], 0)
            self.assertEqual(saved["history"], [])
            result = trainer.train_nn(work, resume=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["history"][0]["loss"], 2.0)
        self.assertEqual(result["progress"]["completed_epochs"], 1)

    def test_phase3_split_matches_unsplit(self):
        vocab = self.root / "tokens.vocab"
        vocab.write_text("<unk>\t0\n<s>\t0\n</s>\t0\n<n>\t0\na\t0\nb\t0\n")
        normalized = self.root / "normalized.json"
        normalized.write_text(json.dumps({"sentences": [{"original": "doc", "lines": ["ababababab"]}]}))
        context = self.root / "input.json"
        context.write_text(json.dumps({"phase1_preprocess": {"normalized_file": str(normalized)},
                                       "phase2_tokenize": {"vocab_file": str(vocab)}}))
        meta = run_phase3_featurize(context, out_directory=self.root / "features", shard_size=3,
                                   context_length=2, shuffle_documents=False,
                                   out_meta_file=self.root / "meta.json", out_context_file=self.root / "output.json")
        xs, ys = resolve_phase3_xy_files(json.loads((self.root / "output.json").read_text()))
        x = sp.vstack([sp.load_npz(p) for p in xs])
        y = np.concatenate([np.load(p) for p in ys])
        expected_x, expected_y, _ = features.build_xy_from_file(normalized, features.VocabTokenizer(vocab), context_length=2)
        self.assertEqual((x != expected_x).nnz, 0)
        np.testing.assert_array_equal(y, expected_y)
        self.assertGreater(len(xs), 1)
        self.assertTrue(all(s["num_samples"] <= 3 for s in meta["outputs"]["shards"]))
        self.assertEqual(xs[0].name, "x_0001.npz")

    def test_actual_training_and_reload(self):
        work = self.root / "actual"
        result = trainer.train_nn(work, x_files=self.xs, y_files=self.ys, epochs=1,
                                  hidden_units=(2,), dropout_rate=0, batch_size=2)
        self.assertEqual(result["stats"]["num_samples"], 4)
        self.assertEqual(result["progress"]["completed_epochs"], 1)
        resumed = trainer.train_nn(work, resume=True)
        self.assertEqual(resumed["history"], result["history"])

    def test_legacy_and_invalid_lists(self):
        xs, ys = resolve_phase3_xy_files({"phase3_featurize": {"x_file": self.xs[0], "y_file": self.ys[0]}})
        self.assertEqual(len(xs), 1)
        with self.assertRaises(ValueError):
            resolve_phase3_xy_files({"phase3_featurize": {"x_files": self.xs, "y_files": self.ys[:1]}})
        with self.assertRaises(ValueError):
            trainer.train_nn(self.root / "bad", x_files=self.xs, y_files=self.ys[:1])


if __name__ == "__main__":
    unittest.main()
