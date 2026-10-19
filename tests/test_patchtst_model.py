from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional locally")
class PatchTSTModelTests(unittest.TestCase):
    def test_forward_shape_and_masked_loss(self) -> None:
        import torch
        from torch import nn

        from trajectory_prediction.models.patchtst import load_patchtst_config
        from trajectory_prediction.models.patchtst.loss import masked_huber_components
        from trajectory_prediction.models.patchtst.model import TrajectoryPatchTST
        from trajectory_prediction.models.patchtst.runtime import resolve_runtime_device

        config = load_patchtst_config(ROOT / "configs/models/patchtst_raw_v1.yaml")
        self.assertEqual(config.implementation_version, "reference_aligned_patchtst_backbone_v1")
        self.assertEqual(config.input_transform, "none")
        self.assertEqual(config.smooth_channels, ())
        self.assertEqual(config.encoder_normalization, "BatchNorm")
        self.assertTrue(config.residual_attention)
        self.assertFalse(config.pre_norm)
        model = TrajectoryPatchTST(config)
        self.assertEqual(len(model.backbone.encoder.encoder.layers), 3)
        self.assertIsInstance(
            model.backbone.encoder.encoder.layers[0].attention_norm.normalization,
            nn.BatchNorm1d,
        )
        history = torch.randn(2, 60, 3)
        prediction = model(history)
        self.assertEqual(tuple(prediction.shape), (2, 60, 3))
        self.assertTrue(torch.isfinite(prediction).all())
        mask = torch.ones(2, 60, dtype=torch.bool)
        mask[:, -1] = False
        loss_sum, count = masked_huber_components(
            prediction, torch.zeros_like(prediction), mask, 1.0, torch
        )
        self.assertTrue(torch.isfinite(loss_sum))
        self.assertEqual(int(count), 2 * 59 * 3)
        runtime = resolve_runtime_device("auto", 0, True, torch)
        self.assertEqual(runtime.resolved, "cpu")
        self.assertFalse(runtime.mixed_precision)

    def test_causal_smooth_config_keeps_the_same_backbone(self) -> None:
        from trajectory_prediction.models.patchtst import load_patchtst_config

        raw = load_patchtst_config(ROOT / "configs/models/patchtst_raw_v1.yaml")
        smooth = load_patchtst_config(
            ROOT / "configs/models/patchtst_causal_smooth_v1.yaml"
        )
        self.assertEqual(smooth.input_transform, "causal_ema")
        self.assertEqual(smooth.smooth_channels, ("up",))
        self.assertEqual(smooth.ema_alpha, 0.4)
        self.assertTrue(smooth.smooth_recenter_at_origin)
        self.assertEqual(smooth.implementation_version, raw.implementation_version)
        self.assertEqual(smooth.model_dimension, raw.model_dimension)
        self.assertEqual(smooth.encoder_layers, raw.encoder_layers)


if __name__ == "__main__":
    unittest.main()
