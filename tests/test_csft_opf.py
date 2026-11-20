from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import CSFT_utils
from actions import ComplexScalarAction
from csft_opf.cli import build_parser, validate_args
from csft_opf.persistence import ensemble_filename, ensemble_files_for_dimension
from csft_opf.optimization import (
    fit_lattice_laplace_profile,
    mean_link_coefficient,
    saddle_point_gradient,
    saddle_point_objective,
    solve_lattice_laplace_field,
    solve_saddle_point_field,
)
from csft_opf.training import make_model
from observables import ComplexScalarOnePtFn
from training import periodic_box_mask


class ComplexScalarActionTests(unittest.TestCase):
    def test_polar_action_matches_cartesian_action(self) -> None:
        generator = torch.Generator().manual_seed(11)
        real = torch.randn(7, 3, 4, generator=generator, dtype=torch.float64)
        imag = torch.randn(7, 3, 4, generator=generator, dtype=torch.float64)
        phi = torch.complex(real, imag)
        radial = phi.abs()
        theta = phi.angle()
        action = ComplexScalarAction(am=1.25, ndims=2)
        polar = action((radial, theta))
        cartesian = 0.5 * 1.25**2 * phi.abs().square().sum((1, 2))
        for axis in (1, 2):
            difference = torch.roll(phi, -1, dims=axis) - phi
            cartesian += 0.5 * difference.abs().square().sum((1, 2))
        torch.testing.assert_close(polar, cartesian)

    def test_deformed_estimator_uses_holomorphic_polar_action(self) -> None:
        generator = torch.Generator().manual_seed(12)
        fields = torch.complex(
            torch.randn(8, 5, generator=generator),
            torch.randn(8, 5, generator=generator),
        )
        model = make_model(fields, ndim=1)
        action = ComplexScalarAction(am=1.0, ndims=1)
        observable = ComplexScalarOnePtFn((5,), (0,), action)
        deformation = torch.tensor(
            [[0.4], [0.2], [0.0], [0.0], [0.2]], dtype=torch.float32
        )
        estimate = observable(model, deform=True, alphas=deformation)
        radial = fields.abs()
        theta = fields.angle()
        delta_action = action((radial, theta + 1j * deformation[..., 0])) - action(
            (radial, theta)
        )
        expected = (
            torch.exp(-deformation[0, 0])
            * fields[:, 0]
            * torch.exp(-delta_action)
        )
        torch.testing.assert_close(estimate, expected)


class ComplexScalarLoadingTests(unittest.TestCase):
    def test_loader_accepts_raw_and_metadata_wrapped_ensembles(self) -> None:
        fields = np.ones((4, 3), dtype=np.complex128) * (1.0 + 2.0j)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.npy"
            wrapped = Path(directory) / "wrapped.npy"
            np.save(raw, fields)
            np.save(wrapped, {"phi": fields, "am": 1.0}, allow_pickle=True)
            raw_fields, raw_mass = CSFT_utils.load_ensemble(raw, expected_am=1.0)
            wrapped_fields, wrapped_mass = CSFT_utils.load_ensemble(
                wrapped, expected_am=1.0
            )
        self.assertIsNone(raw_mass)
        self.assertEqual(wrapped_mass, 1.0)
        self.assertEqual(raw_fields.dtype, torch.complex64)
        torch.testing.assert_close(raw_fields, wrapped_fields)

    def test_loader_accepts_native_generator_npz_archive(self) -> None:
        """Matches the ``Nd{d}_L{L}_m{am:.2f}.npz`` files produced one
        directory above this repository by ``theory.ScalarAction.sample``."""
        rng = np.random.default_rng(0)
        cfgs = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "Nd1_L3_m0.50.npz"
            np.savez(
                archive,
                cfgs=cfgs,
                actions=np.zeros(4),
                Nd=1,
                L=3,
                m=0.5,
                m2=0.25,
            )
            loaded_fields, loaded_mass = CSFT_utils.load_ensemble(
                archive, expected_am=0.5
            )
        self.assertEqual(loaded_mass, 0.5)
        torch.testing.assert_close(
            loaded_fields, torch.as_tensor(cfgs, dtype=torch.complex64)
        )


class ComplexScalarApproximationTests(unittest.TestCase):
    def test_saddle_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(13)
        field = rng.normal(scale=0.2, size=(4, 3))
        coefficient = 0.37
        numerical = np.zeros_like(field)
        epsilon = 1e-6
        for index in np.ndindex(field.shape):
            upper = field.copy()
            lower = field.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            numerical[index] = (
                saddle_point_objective(upper, coefficient)
                - saddle_point_objective(lower, coefficient)
            ) / (2 * epsilon)
        np.testing.assert_allclose(
            saddle_point_gradient(field, coefficient),
            numerical,
            rtol=2e-6,
            atol=2e-7,
        )

    def test_approximation_solvers_preserve_mask_constraints(self) -> None:
        shape = (6, 6)
        source = (0, 0)
        support = periodic_box_mask(shape, source, radius=1).numpy()
        saddle, saddle_diagnostics = solve_saddle_point_field(
            shape,
            source,
            0.4,
            maxiter=1000,
            mask_radius=1,
            gtol=1e-9,
            ftol=1e-12,
        )
        laplace, laplace_diagnostics = solve_lattice_laplace_field(
            shape, source, 0.4, mask_radius=1
        )
        self.assertTrue(saddle_diagnostics["success"])
        self.assertTrue(laplace_diagnostics["success"])
        self.assertEqual(saddle[source], 1.0)
        self.assertEqual(laplace[source], 1.0)
        self.assertTrue(np.all(saddle[~support] == 0.0))
        self.assertTrue(np.all(laplace[~support] == 0.0))

    def test_laplace_profile_is_scored_with_exact_scalar_variance(self) -> None:
        generator = torch.Generator().manual_seed(14)
        fields = torch.complex(
            torch.randn(32, 6, generator=generator),
            torch.randn(32, 6, generator=generator),
        )
        # Add nearest-neighbour correlation so the measured C_phi is positive.
        fields = fields + 0.6 * torch.roll(fields, 1, dims=1)
        self.assertGreater(mean_link_coefficient(fields), 0.0)
        result = fit_lattice_laplace_profile(
            fields,
            am=1.0,
            source=(0,),
            mask_radius=1,
            alpha_bounds=(-2.0, 2.0),
            alpha_grid_points=9,
        )
        self.assertTrue(np.isfinite(result["variance_ratio"]))
        self.assertEqual(np.asarray(result["f_x"])[0, 0], 1.0)


class ComplexScalarPersistenceTests(unittest.TestCase):
    @staticmethod
    @contextlib.contextmanager
    def _data_directory(names: tuple[str, ...]):
        """Point the persistence layer at a directory of empty ensembles."""
        import csft_opf.persistence as persistence_module

        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory)
            for name in names:
                (data_directory / name).touch()
            original = persistence_module.DATA_DIRECTORY
            persistence_module.DATA_DIRECTORY = data_directory
            try:
                yield data_directory
            finally:
                persistence_module.DATA_DIRECTORY = original

    def test_ensemble_discovery_finds_generator_files(self) -> None:
        names = tuple(f"Nd3_L16_m{am:.2f}.npz" for am in (0.33, 0.67, 1.0))
        with self._data_directory(names):
            found = ensemble_files_for_dimension(3, "16")
        self.assertEqual([am for am, _ in found], [0.33, 0.67, 1.0])

    def test_ensemble_filename_matches_generator_convention(self) -> None:
        with self._data_directory(("Nd3_L16_m0.67.npz",)):
            self.assertEqual(
                ensemble_filename(3, 16, 0.67).name, "Nd3_L16_m0.67.npz"
            )

    def test_ensemble_filename_handles_three_decimal_masses(self) -> None:
        """The generator writes as many decimals as the mass needs, so
        ``0.125`` must not be looked up as a two-decimal ``0.12``."""
        names = ("Nd2_L32_m0.125.npz", "Nd2_L32_m0.25.npz")
        with self._data_directory(names):
            self.assertEqual(
                ensemble_filename(2, 32, 0.125).name, "Nd2_L32_m0.125.npz"
            )
            self.assertEqual(
                ensemble_filename(2, 32, 0.25).name, "Nd2_L32_m0.25.npz"
            )
            with self.assertRaises(FileNotFoundError):
                ensemble_filename(2, 32, 0.4)


class ComplexScalarCliTests(unittest.TestCase):
    def test_cli_uses_mass_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["--dims", "1", "--Ls", "8", "--ams", "1.0"]
        )
        validate_args(parser, args)
        self.assertEqual(args.ams, [1.0])
        self.assertFalse(hasattr(args, "betas"))


if __name__ == "__main__":
    unittest.main()
