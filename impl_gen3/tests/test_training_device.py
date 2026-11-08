"""デバイス選択は GPU のない環境でも TensorFlow 設定 API を模擬して検証する。"""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "libs" / "nn_trainer"))
import nn_trainer as trainer


class TrainingDeviceTests(unittest.TestCase):
    def test_cpu_hides_gpus(self):
        with patch.object(trainer.tf.config, "get_visible_devices", return_value=[object()]), \
             patch.object(trainer.tf.config, "set_visible_devices") as hide, \
             patch.object(trainer, "configure_gpu_memory_growth") as growth:
            self.assertEqual(trainer.configure_training_device("cpu"), [])
            hide.assert_called_once_with([], "GPU")
            growth.assert_not_called()

    def test_cpu_already_selected(self):
        with patch.object(trainer.tf.config, "get_visible_devices", return_value=[]), \
             patch.object(trainer.tf.config, "set_visible_devices") as hide:
            self.assertEqual(trainer.configure_training_device("cpu"), [])
            hide.assert_not_called()

    def test_auto_and_gpu(self):
        with patch.object(trainer, "configure_gpu_memory_growth", return_value=["GPU:0"]):
            for mode in ("auto", "gpu"):
                self.assertEqual(trainer.configure_training_device(mode), ["GPU:0"])
        with patch.object(trainer, "configure_gpu_memory_growth", return_value=[]):
            self.assertEqual(trainer.configure_training_device("auto"), [])
            with self.assertRaisesRegex(RuntimeError, "GPU がありません"):
                trainer.configure_training_device("gpu")

    def test_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            trainer.configure_training_device("invalid")
        with patch.object(trainer.tf.config, "get_visible_devices", return_value=[object()]), \
             patch.object(trainer.tf.config, "set_visible_devices", side_effect=RuntimeError("initialized")):
            with self.assertRaisesRegex(RuntimeError, "新しいプロセス"):
                trainer.configure_training_device("cpu")


class TrainingDeviceCLITests(unittest.TestCase):
    def test_resume_passes_device_from_both_clis(self):
        src = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(src / "pipelines" / "training" / "phase4_training"))
        metadata = {
            "stats": {"latest_loss": 1., "latest_perplexity": 2.,
                      "latest_accuracy": 0.5, "peak_rss_mb": 1.},
            "progress": {"stopped_reason": "done", "completed_epochs": 1, "target_epochs": 1},
            "status": "completed", "elapsed_seconds": 1.,
        }
        for module_name, entry in (("nn_trainer.__main__", "train_nn"),
                                   ("phase4_training.__main__", "run_phase4_training")):
            module = importlib.import_module(module_name)
            for device in ("cpu", "gpu", "auto"):
                with self.subTest(module=module_name, device=device), \
                     patch.object(module, entry, return_value=metadata) as train, \
                     patch("builtins.print"):
                    result = module.main(["--work-directory", "/tmp/resume-device-test",
                                          "--resume", f"--device={device}"])
                    self.assertEqual(result, 0)
                    self.assertTrue(train.call_args.kwargs["resume"])
                    self.assertEqual(train.call_args.kwargs["device"], device)

    def test_resume_batch_size_is_optional_in_both_clis(self):
        src = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(src / "pipelines" / "training" / "phase4_training"))
        for module_name, entry in (("nn_trainer.__main__", "train_nn"),
                                   ("phase4_training.__main__", "run_phase4_training")):
            module = importlib.import_module(module_name)
            for batch_size in (None, 2048):
                args = ["--work-directory", "/tmp/resume-batch-test", "--resume"]
                if batch_size is not None:
                    args.append(f"--batch-size={batch_size}")
                with self.subTest(module=module_name, batch_size=batch_size), \
                     patch.object(module, entry, side_effect=RuntimeError("entered training")) as train:
                    with self.assertRaisesRegex(RuntimeError, "entered training"):
                        module.main(args)
                    self.assertTrue(train.call_args.kwargs["resume"])
                    self.assertEqual(train.call_args.kwargs["batch_size"], batch_size)
