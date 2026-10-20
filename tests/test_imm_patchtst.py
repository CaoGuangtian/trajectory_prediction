from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trajectory_prediction.models.imm_patchtst import load_imm_patchtst_config
from trajectory_prediction.models.imm_patchtst.cache import (
    build_corpus_for_split,
    build_imm_prediction_cache,
    open_imm_prediction_cache,
)
from trajectory_prediction.models.patchtst.data import PatchTSTSource


ROOT = Path(__file__).resolve().parents[1]


class ImmPatchTSTCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.parquet_path = self.directory / "standardized.parquet"
        rows: list[dict[str, object]] = []
        for split_index, split in enumerate(("train", "validation", "test")):
            for point in range(125):
                rows.append(
                    {
                        "aircraft": 200 + split_index,
                        "trajectory_id": 2000 + split_index,
                        "segment_id": 20000 + split_index,
                        "point_index": point,
                        "timeAtServer": float(split_index * 1000 + point),
                        "longitude": 10.0 + point * 0.0001,
                        "latitude": 50.0 + point * 0.00005,
                        "geoAltitude": 1000.0 + point * 2.0,
                        "trusted": "unknown",
                        "split": split,
                        "is_interpolated": point == 10,
                        "is_speed_suspicious": point == 20,
                    }
                )
        pq.write_table(pa.Table.from_pylist(rows), self.parquet_path)
        loaded = load_imm_patchtst_config(
            ROOT / "configs/models/imm_patchtst_residual_v1.yaml"
        )
        self.config = replace(loaded, cache_root=self.directory / "cache")
        self.source = PatchTSTSource(1, self.parquet_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_config_composes_matching_patchtst_and_standard_imm(self) -> None:
        self.assertEqual(self.config.fusion_mode, "residual")
        self.assertEqual(
            self.config.patchtst.history_points, self.config.imm.history_seconds
        )
        self.assertEqual(self.config.patchtst.prediction_points, 60)

    def test_cache_is_aligned_and_reopenable(self) -> None:
        cache_dir = self.directory / "cache" / "train"
        metadata = build_imm_prediction_cache(
            [self.source], "train", self.config, cache_dir
        )
        corpus = build_corpus_for_split([self.source], "train", self.config)
        cache = open_imm_prediction_cache(
            cache_dir, corpus, "train", self.config
        )

        self.assertEqual(metadata["window_count"], len(corpus))
        self.assertEqual(cache.positions_enu.shape, (len(corpus), 60, 3))
        self.assertEqual(cache.issue_model_probabilities.shape, (len(corpus), 4))
        self.assertEqual(cache.fallback_mask.shape, (len(corpus),))
        self.assertTrue(np.isfinite(cache.positions_enu).all())
        np.testing.assert_allclose(
            cache.issue_model_probabilities.sum(axis=1), 1.0, atol=1e-5
        )

        mismatched = replace(self.config, config_hash="mismatched")
        with self.assertRaisesRegex(ValueError, "fusion_config_hash"):
            open_imm_prediction_cache(cache_dir, corpus, "train", mismatched)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional locally")
class ImmPatchTSTModelTests(unittest.TestCase):
    def test_zero_residual_returns_exact_imm_prediction(self) -> None:
        import torch
        from torch import nn

        from trajectory_prediction.models.imm_patchtst.model import (
            ImmPatchTSTResidual,
        )

        config = load_imm_patchtst_config(
            ROOT / "configs/models/imm_patchtst_residual_v1.yaml"
        )

        class ZeroResidual(nn.Module):
            def forward(self, history: torch.Tensor) -> torch.Tensor:
                return torch.zeros(
                    history.shape[0],
                    config.patchtst.prediction_points,
                    3,
                    dtype=history.dtype,
                    device=history.device,
                )

        model = ImmPatchTSTResidual(config, residual_model=ZeroResidual())
        history = torch.randn(2, 60, 3)
        imm_prediction = torch.randn(2, 60, 3)
        prediction = model(history, imm_prediction)
        torch.testing.assert_close(prediction, imm_prediction)


if __name__ == "__main__":
    unittest.main()
