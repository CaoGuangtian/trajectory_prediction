from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from trajectory_prediction.models.patchtst import load_patchtst_config
from trajectory_prediction.models.patchtst.data import (
    PatchTSTSource,
    TrajectoryWindowCorpus,
    compute_normalization_scales,
    load_normalization_scales,
    save_normalization_scales,
    transform_history_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


class PatchTSTDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.parquet_path = self.directory / "standardized.parquet"
        rows: list[dict[str, object]] = []
        for split_index, split in enumerate(("train", "validation", "test")):
            for point in range(125):
                rows.append(
                    {
                        "aircraft": 100 + split_index,
                        "trajectory_id": 1000 + split_index,
                        "segment_id": 10000 + split_index,
                        "point_index": point,
                        "timeAtServer": float(split_index * 1000 + point),
                        "longitude": 10.0 + point * 0.0001,
                        "latitude": 50.0 + point * 0.00005,
                        "geoAltitude": 1000.0 + point * 2.0,
                        "trusted": "unknown",
                        "split": split,
                        "is_interpolated": False,
                        "is_speed_suspicious": point == 119,
                    }
                )
        pq.write_table(pa.Table.from_pylist(rows), self.parquet_path)
        self.config = load_patchtst_config(
            ROOT / "configs/models/patchtst_raw_v1.yaml"
        )
        self.smooth_config = load_patchtst_config(
            ROOT / "configs/models/patchtst_causal_smooth_v1.yaml"
        )
        self.source = PatchTSTSource(1, self.parquet_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_windows_and_local_coordinates(self) -> None:
        corpus = TrajectoryWindowCorpus(self.source_list, "train", self.config, 1)
        self.assertEqual(len(corpus), 6)
        batch = corpus.batch([0])
        self.assertEqual(batch.history_enu.shape, (1, 60, 3))
        self.assertEqual(batch.future_enu.shape, (1, 60, 3))
        np.testing.assert_allclose(batch.history_enu[0, -1], 0.0, atol=1e-5)
        self.assertGreater(batch.future_enu[0, 0, 0], 0.0)
        self.assertEqual(int(batch.origin_indices[0]), 59)
        self.assertFalse(batch.target_mask[0, 59])

    def test_split_isolation_and_train_only_normalization(self) -> None:
        train = TrajectoryWindowCorpus(self.source_list, "train", self.config, 2)
        validation = TrajectoryWindowCorpus(
            self.source_list, "validation", self.config, 2
        )
        self.assertTrue(all(segment.split == "train" for segment in train.segments))
        self.assertTrue(
            all(segment.split == "validation" for segment in validation.segments)
        )
        scales = compute_normalization_scales(train, maximum_origins=3, minimum_scale_m=1.0)
        self.assertEqual(scales.fitted_split, "train")
        self.assertTrue(np.all(scales.array >= 1.0))
        path = self.directory / "normalization.json"
        save_normalization_scales(path, scales)
        np.testing.assert_allclose(load_normalization_scales(path).array, scales.array)
        with self.assertRaisesRegex(ValueError, "train split"):
            compute_normalization_scales(
                validation, maximum_origins=3, minimum_scale_m=1.0
            )

    def test_origin_quality_filter_and_trusted_filter(self) -> None:
        config = replace(self.config, trusted_values=("true",))
        corpus = TrajectoryWindowCorpus(self.source_list, "test", config, 1)
        self.assertEqual(len(corpus), 0)

    def test_causal_ema_smooths_only_up_and_recenters_origin(self) -> None:
        history = np.asarray(
            [[[1.0, 2.0, 0.0], [3.0, 4.0, 10.0], [5.0, 6.0, 0.0], [7.0, 8.0, 10.0]]],
            dtype=np.float32,
        )
        config = replace(self.smooth_config, ema_alpha=0.5)
        transformed = transform_history_inputs(history, config)

        np.testing.assert_array_equal(transformed[:, :, :2], history[:, :, :2])
        np.testing.assert_allclose(
            transformed[0, :, 2],
            np.asarray([-6.25, -1.25, -3.75, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(history[0, :, 2], [0.0, 10.0, 0.0, 10.0])

    def test_smooth_variant_changes_only_history_input(self) -> None:
        raw_corpus = TrajectoryWindowCorpus(self.source_list, "train", self.config, 1)
        smooth_corpus = TrajectoryWindowCorpus(
            self.source_list, "train", self.smooth_config, 1
        )
        raw_batch = raw_corpus.batch([0])
        smooth_batch = smooth_corpus.batch([0])

        np.testing.assert_allclose(
            smooth_batch.history_enu[:, :, :2], raw_batch.history_enu[:, :, :2]
        )
        self.assertFalse(
            np.allclose(
                smooth_batch.history_enu[:, :, 2], raw_batch.history_enu[:, :, 2]
            )
        )
        np.testing.assert_allclose(smooth_batch.history_enu[:, -1], 0.0, atol=1e-5)
        np.testing.assert_array_equal(smooth_batch.future_enu, raw_batch.future_enu)
        np.testing.assert_array_equal(smooth_batch.target_mask, raw_batch.target_mask)

    def test_smoothing_does_not_change_normalization_scales(self) -> None:
        raw_corpus = TrajectoryWindowCorpus(self.source_list, "train", self.config, 1)
        smooth_corpus = TrajectoryWindowCorpus(
            self.source_list, "train", self.smooth_config, 1
        )
        raw_scales = compute_normalization_scales(raw_corpus, 4, 1.0)
        smooth_scales = compute_normalization_scales(smooth_corpus, 4, 1.0)
        np.testing.assert_array_equal(raw_scales.array, smooth_scales.array)

    @property
    def source_list(self) -> list[PatchTSTSource]:
        return [self.source]


if __name__ == "__main__":
    unittest.main()
