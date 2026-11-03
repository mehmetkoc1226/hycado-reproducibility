"""Opt-in CPU training/resume integration; never part of a user's normal setup."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("HYCADO_CPU_INTEGRATION") == "1", "opt-in CPU training integration")
class RealTrainerResumeTest(unittest.TestCase):
    def test_interrupted_epoch_resume_matches_uninterrupted_fp32_training(self):
        import json
        import shutil
        import torch
        from ultralytics import YOLO
        from hycado_start import stage1
        from hycado_start.common import sha256, write_json
        from hycado_start.data import prepare, _write_dataset_yaml
        from hycado_start.experiment_data import build_research_data
        from test_start import make_raw

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.set_num_threads(2)
            torch.manual_seed(4)
            # Tiny random architecture is a synthetic harness, not a HyCADO run.
            toy = YOLO("yolov8n.yaml")
            weights = root/"work/fresh_official_pretrained/yolov9c.pt"
            weights.parent.mkdir(parents=True)
            toy.save(weights)
            write_json(weights.with_suffix(".provenance.json"), {"sha256": sha256(weights), "synthetic_test": True})
            del toy
            make_raw(root/"raw", train_count=19, val_count=2)
            prepare(root/"raw", root/"prepared", allow_small=True)
            build_research_data(root/"prepared", root/"dataset", allow_small=True)
            for name in ["continuous", "resumed"]:
                shutil.copytree(root/"dataset", root/name/"data")
                _write_dataset_yaml(root/name/"data/dataset.yaml", root/name/"data")
            config = stage1.protocol()
            config.update(epochs=2, imgsz=64, mosaic=0., close_mosaic=0,
                          cpu_threads=2, implementation="synthetic CPU harness only")
            digest = stage1.fingerprint(config)
            with patch.object(stage1, "ROOT", root):
                full = stage1._train(root/"continuous", config, digest, None, _cpu_validation=True)
                self.assertEqual(full["completed_epochs"], 2)
                real_commit = stage1.commit_checkpoint
                def interrupt_after_first(*args, **kwargs):
                    item = real_commit(*args, **kwargs)
                    if item["epoch"] == 1:
                        raise InterruptedError("synthetic interruption after checkpoint commit")
                    return item
                with patch.object(stage1, "commit_checkpoint", side_effect=interrupt_after_first):
                    with self.assertRaises(InterruptedError):
                        stage1._train(root/"resumed", config, digest, None, _cpu_validation=True)
                checkpoint = stage1.resume_path(root/"resumed", digest)
                resumed = stage1._train(root/"resumed", config, digest, checkpoint, _cpu_validation=True)
                self.assertEqual(resumed["completed_epochs"], 2)
            a = torch.load(stage1.resume_path(root/"continuous", digest), map_location="cpu", weights_only=False)
            b = torch.load(stage1.resume_path(root/"resumed", digest), map_location="cpu", weights_only=False)
            for key in ["model", "ema"]:
                for name, value in a[key].state_dict().items():
                    torch.testing.assert_close(value, b[key].state_dict()[name], rtol=0, atol=0)
            for key, value in a["optimizer"]["state"].items():
                for field, tensor in value.items():
                    if torch.is_tensor(tensor):
                        self.assertEqual(tensor.dtype, b["optimizer"]["state"][key][field].dtype)
                        torch.testing.assert_close(tensor, b["optimizer"]["state"][key][field], rtol=0, atol=0)
            self.assertEqual(a["actual_optimizer_steps"], b["actual_optimizer_steps"])
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["scaler"], b["scaler"])


if __name__ == "__main__":
    unittest.main()
