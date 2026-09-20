import unittest

import numpy as np
import torch

from XY_utils import autocorrelation, integrated_autocorrelation_time


class XYUtilsTests(unittest.TestCase):
    def test_autocorrelation_matches_direct_biased_estimator(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, -1.0, 3.0, 0.5], dtype=torch.float64
        )
        centred = values - values.mean()
        expected = torch.stack(
            [
                torch.dot(centred[: len(values) - lag], centred[lag:])
                / torch.dot(centred, centred)
                for lag in range(len(values))
            ]
        )

        lags, actual = autocorrelation(values)

        self.assertTrue(torch.equal(lags, torch.arange(len(values))))
        self.assertTrue(torch.allclose(actual, expected))

    def test_autocorrelation_pools_independent_chains(self) -> None:
        chains = torch.tensor(
            [[1.0, 0.0, -1.0, 0.5], [3.0, 2.0, 4.0, 1.0]],
            dtype=torch.float64,
        )
        centred = chains - chains.mean(dim=1, keepdim=True)
        expected_lag_one = (
            (centred[:, :-1] * centred[:, 1:]).sum() / centred.square().sum()
        )

        _, actual = autocorrelation(chains, max_lag=1)

        self.assertAlmostEqual(actual[1].item(), expected_lag_one.item())

    def test_integrated_time_for_ar1_process(self) -> None:
        rng = np.random.default_rng(1234)
        phi = 0.8
        innovations = rng.normal(size=(8, 20_000))
        chains = np.empty_like(innovations)
        chains[:, 0] = innovations[:, 0] / np.sqrt(1 - phi**2)
        for time in range(1, chains.shape[1]):
            chains[:, time] = phi * chains[:, time - 1] + innovations[:, time]

        _, acf = autocorrelation(chains)
        actual = integrated_autocorrelation_time(acf)
        expected = (1 + phi) / (2 * (1 - phi))

        self.assertAlmostEqual(actual, expected, delta=0.1 * expected)

    def test_integrated_time_uses_positive_monotone_pairs(self) -> None:
        # The isolated negative rho[2] should not truncate the sum by itself.
        acf = torch.tensor([1.0, 0.6, -0.05, 0.4, -0.2, -0.3])

        self.assertAlmostEqual(integrated_autocorrelation_time(acf), 1.45)

    def test_constant_series_has_no_estimable_integrated_time(self) -> None:
        _, acf = autocorrelation(torch.ones(20))

        self.assertTrue(torch.isnan(acf).all())
        with self.assertRaisesRegex(ValueError, "finite"):
            integrated_autocorrelation_time(acf)


if __name__ == "__main__":
    unittest.main()
