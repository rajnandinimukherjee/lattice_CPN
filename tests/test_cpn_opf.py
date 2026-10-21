import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from CPN_utils import ComplexPair
from actions import CPnGaugeAction, CPnQuarticAction
from cpn_opf.cli import build_parser, validate_args
from cpn_opf.loading import load_ensemble
from cpn_opf.model import CPnHomogeneousDeformations
from cpn_opf.objectives import (
    init_variance_network,
    train_penalty_loss,
    train_variance_loss,
)
from cpn_opf.persistence import ensemble_savename, ratio_savename, training_savename
from cpn_opf.training import make_deformation_mask, penalty_source_constraints
from cpn_opf.plotting import plot_deformation_fields, plot_scaling
from cpn_opf.study import run_analysis
from deformations import HomogeneousDeformations
from networks import UNet
from observables import CPnOnePtFn, cpn_one_pt_fn


class CPNOPFTests(unittest.TestCase):
    def test_show_is_deferred_until_all_analysis_figures_are_created(self) -> None:
        """One final show call keeps summary and field figures visible together."""
        args = SimpleNamespace(
            N=2,
            action="gauge",
            dims=[1],
            Ls=[3],
            loss_function="variance",
            betas=[],
            delete_betas=[],
            run=False,
            save_plot=False,
            show=True,
            plot_field=True,
            i=0,
            j=1,
        )

        def load_dimension(
            _args, _ndim, _lattice_size, loss_functions, ratios, fields, betas
        ):
            for loss_function in loss_functions:
                ratios[loss_function].append(np.asarray([1.0]))
                fields[loss_function].append(np.asarray([0.0]))
                betas[loss_function].append([1.0])

        with (
            patch("cpn_opf.study._load_dimension", side_effect=load_dimension),
            patch("cpn_opf.study.plot_scaling") as scaling_plot,
            patch("cpn_opf.study.training_files_for_dimension", return_value=[]),
            patch("cpn_opf.study.plot_deformation_fields") as field_plot,
            patch("cpn_opf.study.plt.show") as show,
        ):
            run_analysis(args)

        self.assertFalse(scaling_plot.call_args.kwargs["show"])
        self.assertFalse(field_plot.call_args.kwargs["show"])
        show.assert_called_once_with()

    def test_variance_network_is_seeded_away_from_zero_field(self) -> None:
        torch.manual_seed(31)
        network = UNet(input_dim=6, output_dim=3, ndim=1, hd=2)
        init_variance_network(network)

        self.assertTrue(torch.count_nonzero(network.output.weight))
        self.assertTrue(torch.count_nonzero(network.output.bias))
        field = network(torch.zeros(8, 6))
        self.assertGreater(field.abs().max().item(), 0.0)

    def test_cp_specific_artifact_names_include_all_scaling_inputs(self) -> None:
        self.assertEqual(
            ensemble_savename(2, 2, 4.0, 64, "gauge"),
            Path("data/cp2_gauge_dim2_b4.00_L64_ens.npy"),
        )
        self.assertEqual(
            ensemble_savename(2, 2, 4.0, 64, "quartic"),
            Path("data/cp2_quartic_dim2_b4.00_L64_ens.npy"),
        )
        self.assertEqual(
            training_savename(2, 2, 4.0, 64, 0, 1),
            Path(
                "training_data/cp2_scaling_dim2_b4_L64_i0_j1_"
                "action-gauge_loss-variance_training.npy"
            ),
        )
        self.assertEqual(
            ratio_savename(2, 2, 64, 0, 1),
            Path(
                "training_data/cp2_scaling_dim2_L64_i0_j1_"
                "action-gauge_loss-variance_var_ratios.npy"
            ),
        )

    def test_loader_thins_paired_vectors_and_links(self) -> None:
        rng = np.random.default_rng(7)
        n, ndim, length, samples = 1, 1, 3, 24
        z = rng.normal(size=(1, samples, length, n + 1))
        z = z + 1j * rng.normal(size=z.shape)
        z /= np.linalg.vector_norm(z, axis=-1, keepdims=True)
        links = rng.normal(size=(1, samples, length, ndim))

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "cp1_gauge_dim1_b1.00_L3_ens.npy"
            np.save(filename, {"z": z, "theta": links, "beta": 1.0})
            fields, loaded_links, saved_beta = load_ensemble(
                filename,
                n=n,
                ndim=ndim,
                lattice_size=length,
                beta=1.0,
            )

        self.assertEqual(saved_beta, 1.0)
        self.assertEqual(tuple(fields.shape[1:]), (length, 2 * (n + 1)))
        self.assertEqual(tuple(loaded_links.shape[1:]), (length, ndim))
        self.assertEqual(len(fields), len(loaded_links))
        self.assertGreaterEqual(len(fields), 1)

    def test_quartic_loader_accepts_a_vector_only_ensemble(self) -> None:
        rng = np.random.default_rng(11)
        # L=4 makes channel-first and channel-last shapes ambiguous, so this
        # also verifies that a vector-only ``x`` payload keeps its generator
        # layout when no ``lam`` array is present.
        n, ndim, length, samples = 1, 1, 4, 24
        x = rng.normal(size=(samples, 2 * (n + 1), length))
        x /= np.linalg.vector_norm(x, axis=1, keepdims=True)

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "cp1_quartic_dim1_b1.00_L4_ens.npy"
            np.save(filename, {"x": x, "beta": 1.0})
            fields, links, saved_beta = load_ensemble(
                filename,
                n=n,
                ndim=ndim,
                lattice_size=length,
                beta=1.0,
                action_type="quartic",
            )

        self.assertEqual(saved_beta, 1.0)
        self.assertEqual(tuple(fields.shape[1:]), (length, 2 * (n + 1)))
        expected = torch.as_tensor(x[0]).movedim(0, -1)
        expected = torch.stack(
            (expected[..., : n + 1], expected[..., n + 1 :]), dim=-1
        ).flatten(start_dim=-2)
        torch.testing.assert_close(fields[0], expected.to(torch.float32))
        self.assertIsNone(links)

    def test_loader_accepts_split_real_gauge_layout(self) -> None:
        rng = np.random.default_rng(8)
        n, ndim, length, chains, samples = 1, 1, 3, 2, 24
        x = rng.normal(size=(chains, samples, 2 * (n + 1), length))
        x /= np.linalg.vector_norm(x, axis=2, keepdims=True)
        lam = rng.normal(size=(chains, samples, ndim, length))

        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "cp1_gauge_dim1_b1.00_L3_ens.npy"
            np.save(
                filename,
                {"x": x, "lam": lam, "energies": np.zeros(samples), "beta": 1.0},
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fields, loaded_links, saved_beta = load_ensemble(
                    filename,
                    n=n,
                    ndim=ndim,
                    lattice_size=length,
                    beta=1.0,
                )

        expected = torch.as_tensor(x[0, 0]).movedim(0, -1)
        expected = torch.stack(
            (expected[..., : n + 1], expected[..., n + 1 :]), dim=-1
        ).flatten(start_dim=-2)
        torch.testing.assert_close(fields[0], expected.to(torch.float32))
        torch.testing.assert_close(
            loaded_links[0], torch.as_tensor(lam[0, 0]).movedim(0, -1).float()
        )
        self.assertEqual(saved_beta, 1.0)
        self.assertFalse(
            any("ignoring an energies array" in str(warning.message) for warning in caught)
        )

    def test_gauge_action_matches_split_real_convention(self) -> None:
        torch.manual_seed(9)
        n, ndim, length, batch = 1, 2, 3, 2
        x_split = torch.randn(batch, 2 * (n + 1), length, length)
        x_split /= torch.linalg.vector_norm(x_split, dim=1, keepdim=True)
        lam_split = torch.randn(batch, ndim, length, length)

        x_last = x_split.movedim(1, -1)
        fields = torch.stack(
            (x_last[..., : n + 1], x_last[..., n + 1 :]), dim=-1
        ).flatten(start_dim=-2)
        lam_last = lam_split.movedim(1, -1)

        expected = torch.zeros(batch)
        omega_x = torch.cat(
            (-x_split[:, n + 1 :], x_split[:, : n + 1]), dim=1
        )
        for direction in range(ndim):
            x_forward = torch.roll(x_split, -1, dims=direction + 2)
            a = (x_forward * x_split).sum(1)
            b = (x_forward * omega_x).sum(1)
            expected -= 2 * (
                a * torch.cos(lam_split[:, direction])
                + b * torch.sin(lam_split[:, direction])
            ).flatten(1).sum(1)

        action = CPnGaugeAction(1.0, n, (length, length))
        torch.testing.assert_close(action.energy((fields, lam_last)), expected)

    def test_local_gauge_energy_change_matches_global_action_change(self) -> None:
        torch.manual_seed(12)
        n, length = 1, 4
        fields = torch.randn(length, length, 2 * (n + 1))
        fields /= torch.linalg.vector_norm(fields, dim=-1, keepdim=True)
        links = torch.randn(length, length, 2)
        proposal = fields.clone()
        proposal[1, 2] = torch.randn(2 * (n + 1))
        proposal[1, 2] /= torch.linalg.vector_norm(proposal[1, 2])
        sites = torch.tensor([[1, 2]])

        action = CPnGaugeAction(1.0, n, (length, length))
        global_delta = action.energy((proposal, links)) - action.energy(
            (fields, links)
        )
        local_delta = action.local_energy(
            (proposal, links), sites
        ) - action.local_energy((fields, links), sites)
        torch.testing.assert_close(global_delta, local_delta.squeeze())

    def test_gauge_observable_evaluates_and_backpropagates(self) -> None:
        torch.manual_seed(2)
        n, ndim, length, configurations = 1, 1, 3, 6
        z = torch.randn(configurations, length, n + 1, dtype=torch.cfloat)
        z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        fields = torch.view_as_real(z).flatten(start_dim=-2)
        links = torch.randn(configurations, length, ndim)
        model = CPnHomogeneousDeformations(n, action_type="gauge")
        mask = torch.tensor([True, True, True, True, False, False])
        model.samples = {
            "all": fields,
            "train": fields[mask],
            "test": fields[~mask],
        }
        model.set_gauge_samples(
            {
                "all": links,
                "train": links[mask],
                "test": links[~mask],
            }
        )
        action = CPnGaugeAction(1.0, n, (length,))
        observable = CPnOnePtFn(
            n, (length,), source=(0,), action=action
        )
        alphas = torch.zeros(length, n + 1, requires_grad=True)
        values = observable(model, deform=True, alphas=alphas, sampletype="all")
        loss = values.real.square().mean()
        loss.backward()

        self.assertIsInstance(values, ComplexPair)
        self.assertFalse(values.real.is_complex())
        self.assertFalse(values.imag.is_complex())
        self.assertTrue(torch.isfinite(values.real).all())
        self.assertIsNotNone(alphas.grad)
        self.assertTrue(torch.isfinite(alphas.grad).all())

    def test_real_jacobian_matches_native_complex_reference(self) -> None:
        """The production Jacobian never needs a backend complex dtype."""
        torch.manual_seed(18)
        for n in range(4):
            model = HomogeneousDeformations(n)
            x = torch.randn(2, 3, 2 * (n + 1), dtype=torch.float64)
            alphas = (
                0.2 * torch.randn(3, n + 1, dtype=torch.float64)
            ).requires_grad_()

            y = model.deformx(x, alphas)
            normalization = (1 + y.square().sum(dim=-1)).sqrt()
            diagonal = torch.diag_embed(
                torch.repeat_interleave(alphas, repeats=2, dim=-1)
            )
            jacobian = normalization[..., None, None] * model.eye.to(x)
            jacobian = jacobian + torch.einsum(
                "...ij,...j,...k->...ik", diagonal @ diagonal, x, x
            ) / normalization[..., None, None]
            reference = torch.linalg.det(
                jacobian.to(torch.complex128)
                - 1j * (diagonal @ model.omega.to(x))
            ) / normalization.square()

            determinant = model.detJac(x, alphas)
            torch.testing.assert_close(determinant, reference.real)
            torch.testing.assert_close(
                reference.imag, torch.zeros_like(reference.imag)
            )
            actual_grad = torch.autograd.grad(
                determinant.sum(), alphas, retain_graph=True
            )[0]
            reference_grad = torch.autograd.grad(reference.real.sum(), alphas)[0]
            torch.testing.assert_close(actual_grad, reference_grad)

    def test_real_pair_matches_native_complex_action_and_observable(self) -> None:
        """Real-pair contour arithmetic is algebraically unchanged on CPU."""
        torch.manual_seed(19)
        n, length = 1, 3
        x = torch.randn(2, length, 2 * (n + 1), dtype=torch.float64)
        alphas = 0.1 * torch.randn(length, n + 1, dtype=torch.float64)
        pair = HomogeneousDeformations(n).deform(x, alphas)
        native = pair.to_complex()
        links = torch.randn(2, length, 1, dtype=torch.float64)

        gauge = CPnGaugeAction(1.2, n, (length,))
        quartic = CPnQuarticAction(1.2, n, (length,))
        torch.testing.assert_close(
            gauge((pair, links)).to_complex(), gauge((native, links))
        )
        torch.testing.assert_close(quartic(pair).to_complex(), quartic(native))
        torch.testing.assert_close(
            cpn_one_pt_fn(pair, 0, 1, (1,)).to_complex(),
            cpn_one_pt_fn(native, 0, 1, (1,)),
        )

    def test_scaling_plot_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "scaling.png"
            figure = plot_scaling(
                [1, 2],
                [8, 4],
                {"variance": [[0.5, 1.0], [0.5, 1.0]]},
                {
                    "variance": [
                        np.asarray([0.9, 0.7]),
                        np.asarray([0.8, 0.6]),
                    ]
                },
                {
                    "variance": [
                        np.asarray([0.1, 0.2]),
                        np.asarray([0.2, 0.3]),
                    ]
                },
                ["variance"],
                n=2,
                i=0,
                j=1,
                filename=filename,
            )
            self.assertTrue(filename.exists())
            ratio_axes = figure.axes[:2]
            field_axes = figure.axes[2:]
            self.assertTrue(
                ratio_axes[0].get_shared_y_axes().joined(*ratio_axes)
            )
            self.assertTrue(
                field_axes[0].get_shared_y_axes().joined(*field_axes)
            )
            figure.clf()

    def test_scaling_plot_uses_temperature_and_marks_critical_point(self) -> None:
        figure = plot_scaling(
            [2],
            [4],
            {"variance": [[0.5, 1.0]]},
            {"variance": [np.asarray([0.8, 0.6])]},
            {"variance": [np.asarray([0.2, 0.3])]},
            ["variance"],
            n=2,
            i=0,
            j=1,
        )
        ratio_ax, field_ax = figure.axes
        np.testing.assert_allclose(ratio_ax.lines[0].get_xdata(), [2.0, 1.0])
        np.testing.assert_allclose(field_ax.lines[0].get_xdata(), [2.0, 1.0])
        self.assertEqual(figure._supxlabel.get_text(), r"$T=\beta^{-1}$")
        critical_axis = ratio_ax.child_axes[0]
        self.assertEqual(
            critical_axis.get_xticklabels()[0].get_text(), r"$T_{\rm c}=0$"
        )
        figure.clf()

    def test_deformation_field_plots_support_1d_2d_and_3d(self) -> None:
        """Every CP(N) channel is rendered for each beta in every dimension."""
        n, length = 2, 3
        with tempfile.TemporaryDirectory() as directory:
            for ndim in (1, 2, 3):
                artifacts = []
                for beta in (0.5, 1.0):
                    field = np.arange(
                        length**ndim * (n + 1), dtype=float
                    ).reshape((length,) * ndim + (n + 1,))
                    filename = Path(directory) / f"d{ndim}_b{beta}.npy"
                    np.save(
                        filename,
                        {
                            "scaled_field": field,
                            "metadata": {"source": (0,) * ndim},
                        },
                        allow_pickle=True,
                    )
                    artifacts.append((beta, filename))
                output = Path(directory) / f"d{ndim}.png"
                figure = plot_deformation_fields(
                    artifacts,
                    n=n,
                    ndim=ndim,
                    lattice_size=length,
                    i=0,
                    j=1,
                    loss_function="variance",
                    filename=output,
                )
                self.assertTrue(output.exists())
                # One axes per beta/channel plus a shared colorbar.
                self.assertEqual(len(figure.axes), 2 * (n + 1) + 1)
                self.assertGreater(figure.axes[-1].get_position().x0, 0.9)
                heatmap_axes = figure.axes[:-1]
                self.assertTrue(
                    heatmap_axes[0].get_shared_x_axes().joined(
                        heatmap_axes[0], heatmap_axes[-1]
                    )
                )
                self.assertTrue(
                    heatmap_axes[0].get_shared_y_axes().joined(
                        heatmap_axes[0], heatmap_axes[-1]
                    )
                )
                self.assertIn(
                    ", loss function: variance", figure._suptitle.get_text()
                )
                self.assertNotIn("\n", figure._suptitle.get_text())
                if ndim > 1:
                    self.assertEqual(
                        figure.axes[0].get_ylabel(), "$\\beta=0.5$"
                    )
                figure.clf()

    def test_one_epoch_training_writes_a_gauge_result(self) -> None:
        torch.manual_seed(3)
        n, ndim, length, configurations = 1, 1, 3, 6
        z = torch.randn(configurations, length, n + 1, dtype=torch.cfloat)
        z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        fields = torch.view_as_real(z).flatten(start_dim=-2)
        links = torch.randn(configurations, length, ndim)
        action = CPnGaugeAction(1.0, n, (length,))
        observable = CPnOnePtFn(
            n, (length,), source=(0,), action=action
        )
        args = SimpleNamespace(
            unet_hidden_dim=2,
            learning_rate=1e-3,
            epochs=1,
            batch_size=2,
            mask_radius=1,
            max_training_time=None,
            plateau_window=2,
            plateau_patience=2,
            plateau_min_delta=1e-3,
            plateau_threshold_mode="rel",
            early_stopping=True,
            lr_scheduler="constant",
            lr_factor=0.5,
            lr_patience=1,
            lr_cooldown=0,
            min_learning_rate=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "result.npy"
            ratio = train_variance_loss(
                fields,
                links,
                n=n,
                action_type="gauge",
                ndim=ndim,
                beta=1.0,
                obs=observable,
                args=args,
                source=(0,),
                i=0,
                j=0,
                training_savename=filename,
                device=torch.device("cpu"),
            )
            data = np.load(filename, allow_pickle=True).item()

        self.assertTrue(np.isfinite(ratio))
        self.assertEqual(data["metadata"]["action_type"], "gauge")
        self.assertEqual(data["metadata"]["cp_n"], n)

    def test_one_epoch_training_writes_a_quartic_result_without_links(self) -> None:
        torch.manual_seed(4)
        n, ndim, length, configurations = 1, 1, 3, 6
        z = torch.randn(configurations, length, n + 1, dtype=torch.cfloat)
        z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        fields = torch.view_as_real(z).flatten(start_dim=-2)
        action = CPnQuarticAction(1.0, n, (length,))
        observable = CPnOnePtFn(
            n, (length,), source=(0,), action=action
        )
        args = SimpleNamespace(
            unet_hidden_dim=2,
            learning_rate=1e-3,
            epochs=1,
            batch_size=2,
            mask_radius=1,
            max_training_time=None,
            plateau_window=2,
            plateau_patience=2,
            plateau_min_delta=1e-3,
            plateau_threshold_mode="rel",
            early_stopping=True,
            lr_scheduler="constant",
            lr_factor=0.5,
            lr_patience=1,
            lr_cooldown=0,
            min_learning_rate=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "result.npy"
            ratio = train_variance_loss(
                fields,
                None,
                n=n,
                action_type="quartic",
                ndim=ndim,
                beta=1.0,
                obs=observable,
                args=args,
                source=(0,),
                i=0,
                j=0,
                training_savename=filename,
                device=torch.device("cpu"),
            )
            data = np.load(filename, allow_pickle=True).item()

        self.assertTrue(np.isfinite(ratio))
        self.assertEqual(data["metadata"]["action_type"], "quartic")

    def test_one_epoch_penalty_training_writes_a_scaled_gauge_result(self) -> None:
        torch.manual_seed(24)
        n, ndim, length, configurations = 1, 1, 3, 6
        z = torch.randn(configurations, length, n + 1, dtype=torch.cfloat)
        z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        fields = torch.view_as_real(z).flatten(start_dim=-2)
        links = torch.randn(configurations, length, ndim)
        action = CPnGaugeAction(0.1, n, (length,))
        observable = CPnOnePtFn(
            n, (length,), i=0, j=1, source=(0,), action=action
        )
        args = SimpleNamespace(
            unet_hidden_dim=2,
            learning_rate=1e-3,
            epochs=1,
            batch_size=2,
            mask_radius=1,
            max_training_time=None,
            plateau_window=2,
            plateau_patience=2,
            plateau_min_delta=1e-3,
            plateau_threshold_mode="rel",
            early_stopping=True,
            lr_scheduler="constant",
            lr_factor=0.5,
            lr_patience=1,
            lr_cooldown=0,
            min_learning_rate=0.0,
            alpha_bounds=(-1.0, 1.0),
            alpha_grid_points=5,
            alpha_xatol=1e-3,
        )
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "penalty.npy"
            ratio = train_penalty_loss(
                fields,
                links,
                n=n,
                action_type="gauge",
                ndim=ndim,
                beta=0.1,
                obs=observable,
                args=args,
                source=(0,),
                i=0,
                j=1,
                training_savename=filename,
                device=torch.device("cpu"),
            )
            data = np.load(filename, allow_pickle=True).item()

        self.assertTrue(np.isfinite(ratio))
        self.assertEqual(data["loss_function"], "penalty")
        self.assertEqual(data["metadata"]["loss_function"], "penalty")
        np.testing.assert_array_equal(data["f_source_components"], [0, 1])
        np.testing.assert_allclose(
            data["f_source_component_values"],
            [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
        )
        self.assertEqual(data["f_source_norm_constraint"], 1.0)
        np.testing.assert_allclose(
            data["learned_field"][0],
            [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(data["learned_field"][0])), 1.0, places=6
        )
        np.testing.assert_allclose(
            data["scaled_field"],
            data["alpha_scale"] * data["learned_field"],
        )

    def test_penalty_source_constraint_is_unit_norm_and_opf_symmetric(self) -> None:
        constraints = penalty_source_constraints(
            source=(2, 3), channels=4, i=1, j=3
        )
        source_vector = np.asarray(
            [constraints[(2, 3, channel)] for channel in range(4)]
        )
        np.testing.assert_allclose(
            source_vector,
            [0.0, 1.0 / np.sqrt(2), 0.0, -1.0 / np.sqrt(2)],
        )
        self.assertAlmostEqual(np.linalg.norm(source_vector), 1.0)

        diagonal = penalty_source_constraints(
            source=(2,), channels=4, i=2, j=2
        )
        np.testing.assert_allclose(
            [diagonal[(2, channel)] for channel in range(4)],
            [0.0, 0.0, 1.0, 0.0],
        )

    def test_cli_defaults_to_gauge_and_accepts_quartic(self) -> None:
        parser = build_parser()
        gauge = parser.parse_args(["--dims", "1", "--Ls", "3"])
        validate_args(parser, gauge)
        self.assertEqual(gauge.action, "gauge")
        self.assertEqual(gauge.mask_radius, 6)
        self.assertFalse(gauge.plot_field)

        quartic = parser.parse_args(
            ["--action", "Quartic", "--dims", "1", "--Ls", "3"]
        )
        validate_args(parser, quartic)
        self.assertEqual(quartic.action, "quartic")

        field_plot = parser.parse_args(
            ["--dims", "1", "--Ls", "3", "--plot-field"]
        )
        validate_args(parser, field_plot)
        self.assertTrue(field_plot.plot_field)

    def test_deformation_mask_zeroes_every_channel_outside_radius(self) -> None:
        mask = make_deformation_mask(
            (8,), source=(0,), radius=1, channels=3
        )

        self.assertEqual(tuple(mask.shape), (8, 3))
        self.assertTrue(mask[0].all())
        self.assertTrue(mask[1].all())
        self.assertTrue(mask[-1].all())
        self.assertFalse(mask[2].any())
        self.assertFalse(mask[-2].any())

    def test_cli_can_select_the_penalty_objective(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--dims", "1", "--Ls", "3", "--loss-function", "penalty",
                "--alpha-bounds", "-2", "3", "--alpha-grid-points", "7",
            ]
        )
        validate_args(parser, args)
        self.assertEqual(args.loss_function, "penalty")
        self.assertEqual(args.alpha_bounds, [-2.0, 3.0])
        self.assertEqual(args.alpha_grid_points, 7)

    def test_quartic_action_uses_vector_fields_without_links(self) -> None:
        torch.manual_seed(16)
        n, length = 1, 3
        fields = torch.randn(2, length, 2 * (n + 1))
        fields /= torch.linalg.vector_norm(fields, dim=-1, keepdim=True)
        action = CPnQuarticAction(1.3, n, (length,))

        self.assertEqual(tuple(action.energy(fields).shape), (2,))
        torch.testing.assert_close(action(fields), 1.3 * action.energy(fields))


if __name__ == "__main__":
    unittest.main()
