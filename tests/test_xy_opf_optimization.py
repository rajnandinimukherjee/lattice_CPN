import unittest

import numpy as np

from xy_opf.optimization import (
    saddle_point_gradient,
    saddle_point_objective,
    solve_saddle_point_field,
)


class XYOPFOptimizationTests(unittest.TestCase):
    def test_saddle_gradient_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(123)
        field = rng.normal(scale=0.2, size=(4, 3))
        beta = 0.7
        epsilon = 1e-6
        numerical = np.empty_like(field)

        for index in np.ndindex(field.shape):
            upper = field.copy()
            lower = field.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            numerical[index] = (
                saddle_point_objective(upper, beta)
                - saddle_point_objective(lower, beta)
            ) / (2 * epsilon)

        np.testing.assert_allclose(
            saddle_point_gradient(field, beta),
            numerical,
            rtol=1e-7,
            atol=1e-8,
        )

    def test_solver_preserves_source_and_localisation_constraints(self) -> None:
        field, diagnostics = solve_saddle_point_field(
            (7,),
            (0,),
            1.0,
            maxiter=200,
            mask_radius=1,
            gtol=1e-10,
            ftol=1e-14,
        )

        self.assertTrue(diagnostics["success"])
        self.assertEqual(field[0], 1.0)
        np.testing.assert_array_equal(field[2:6], 0.0)
        self.assertTrue(np.all((0.0 <= field) & (field <= 1.0)))


if __name__ == "__main__":
    unittest.main()
