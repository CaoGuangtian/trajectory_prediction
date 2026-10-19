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

    @property
    def source_list(self) -> list[PatchTSTSource]:
        return [self.source]


if __name__ == "__main__":
    unittest.main()
