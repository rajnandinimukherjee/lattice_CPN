import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from xy_opf.persistence import (
    delete_beta_training_data,
    load_variance_ratios,
    ratio_savename,
    training_savename,
    upsert_variance_ratios,
)


class XYOPFPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.previous_directory = Path.cwd()
        os.chdir(self.temporary_directory.name)
        Path("training_data").mkdir()

    def tearDown(self) -> None:
        os.chdir(self.previous_directory)
        self.temporary_directory.cleanup()

    def test_upsert_adds_and_replaces_only_selected_betas(self) -> None:
        filename = ratio_savename(2, "variance")
        upsert_variance_ratios(
            filename,
            ndim=2,
            lattice_size=32,
            loss_function="variance",
            updates={1.20: 0.8, 1.30: 0.7},
        )

        betas, ratios = upsert_variance_ratios(
            filename,
            ndim=2,
            lattice_size=32,
            loss_function="variance",
            updates={1.20: 0.5, 1.25: 0.6},
        )

        np.testing.assert_allclose(betas, [1.20, 1.25, 1.30])
        np.testing.assert_allclose(ratios, [0.5, 0.6, 0.7])

    def test_delete_removes_only_requested_beta(self) -> None:
        filename = ratio_savename(2, "penalty")
        upsert_variance_ratios(
            filename,
            ndim=2,
            lattice_size=32,
            loss_function="penalty",
            updates={1.20: 0.8, 1.30: 0.7},
        )
        deleted_artifact = training_savename(2, 1.20, "penalty")
        retained_artifact = training_savename(2, 1.30, "penalty")
        np.save(deleted_artifact, {"variance_ratio": 0.8})
        np.save(retained_artifact, {"variance_ratio": 0.7})

        deleted_files, removed_betas = delete_beta_training_data(
            2, "penalty", [1.20]
        )

        self.assertEqual(deleted_files, [deleted_artifact])
        self.assertEqual(removed_betas, [1.20])
        self.assertFalse(deleted_artifact.exists())
        self.assertTrue(retained_artifact.exists())
        np.testing.assert_allclose(
            load_variance_ratios(filename, 2, [1.30], "penalty"),
            [0.7],
        )

    def test_upsert_recovers_an_artifact_missing_from_the_index(self) -> None:
        filename = ratio_savename(2, "penalty")
        upsert_variance_ratios(
            filename,
            ndim=2,
            lattice_size=32,
            loss_function="penalty",
            updates={1.30: 0.7},
        )
        np.save(
            training_savename(2, 1.20, "penalty"),
            {"variance_ratio": 0.8},
        )

        betas, ratios = upsert_variance_ratios(
            filename,
            ndim=2,
            lattice_size=32,
            loss_function="penalty",
            updates={1.40: 0.6},
        )

        np.testing.assert_allclose(betas, [1.20, 1.30, 1.40])
        np.testing.assert_allclose(ratios, [0.8, 0.7, 0.6])


if __name__ == "__main__":
    unittest.main()
