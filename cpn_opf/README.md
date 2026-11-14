# CP(N) OPF analysis layout

`CPN_OPF_analysis.py` is the entry point for the CP(N) one-point-function
scaling study.  It mirrors the organisation of `xy_opf`:

- `cli.py` defines and validates the command line;
- `loading.py` loads CP(N) ensembles and thins them using integrated
  autocorrelation times;
- `model.py` selects vector-only or vector-plus-gauge-link action inputs;
- `objectives.py` runs direct OPF-variance and penalty-field minimisation;
- `optimization.py` performs Approximation 2 profile and scalar field-scale
  optimization;
- `training.py` contains split, trainer, and persistence helpers;
- `persistence.py` owns artifact names and ratio indexes;
- `plotting.py` writes the scaling figures; and
- `study.py` orchestrates running or reloading a series.

Pass `--no-save-legend` when the scaling plot will use a shared legend in a
larger figure.
For direct-variance refinement from an existing CP(N) Approximation 2 result,
pass `--variance-init-from-approx2`. The run adds a learned U-Net residual to
the saved scaled profile and retains the lower-variance result after a final
scalar-scale search.

Ensembles are `.npy` mappings with complex `z` vectors, interleaved real
coordinates, or copied split real coordinates. Gauge ensembles additionally
store `lam` link angles; the copied layout stores `x` and `lam` as `(chains,
samples, components, *lattice)`, which the loader converts to interleaved
coordinates and `(*lattice, direction)` links. The gauge action is then
evaluated in those real coordinates, including on the deformed contour, with
the same `lam` convention as the generator. Quartic ensembles require only
the vector array. Canonical arrays may instead use `(chains, samples,
*lattice, components)`. The canonical input name is
`data/cp{N}_{action}_dim{ndim}_b{beta:.2f}_L{L}_ens.npy`, for example
`data/cp2_gauge_dim1_b1.50_L64_ens.npy`.

The CP(N) deformation path retains this real representation throughout.
Deformed coordinates are stored internally as a pair of real tensors
(`real`, `imag`), so the action, observable, and analytic Jacobian can run on
MPS without constructing a complex PyTorch tensor. The Jacobian determinant
is evaluated in real arithmetic; it is real for this homogeneous deformation.
Only materialize a native complex tensor for an output boundary that requires
one, using `deformed.cpu().to_complex()`.

Choose the action with `--action Gauge` (the default) or `--action Quartic`.
Quartic analyses do not require link angles; if a vector ensemble also carries
them, they are ignored.

By default the study compares direct `variance` minimisation, `penalty`
minimisation of `<exp(-2 Re Delta S)>`, and the numerical CP(N)
small-fluctuation `approx2` and smooth-deformation `approx3` profiles. The
penalty objective fixes a unit-norm
deformation vector at the source to break the zero-field stationary point:
for `i != j`, the `i` component is `1/sqrt(2)`, the `j` component is
`-1/sqrt(2)`, and every other source component is zero; for `i == j`, the `i`
component is one. It then numerically chooses a scalar `alpha` to minimize
the full-ensemble OPF variance. Approximation 2 solves its small-fluctuation
penalty on the same localized support, then optimizes one overall scale
against the ensemble. Approximation 2 is available for the gauge action.
Approximation 3 solves the nonlinear smooth-deformation metric penalty,
initialized from the Approximation 2 profile, then optimizes its scale against
the full OPF variance. Approximation 3 is available for the gauge action and
uses at most 1,280 decorrelated configurations by default; each beta result
and the ratio index are saved as the scan proceeds. Matching saved results are
reused on a rerun. Use `--recompute-approx3` to replace them, or adjust the
sample cap with `--approx3-max-configurations`.
Use `--loss-function variance`, `--loss-function penalty`,
`--loss-function approx2`, or `--loss-function approx3` to select one
objective. The penalty scale search
accepts `--alpha-bounds LOWER UPPER`, `--alpha-grid-points`, and
`--alpha-xatol`, as in the XY analysis. Approximation 2 accepts
`--approx2-maxiter`, `--approx2-gtol`, and `--approx2-ftol` for its numerical
solve.

Pass `--plot-field` to render the saved, scaled deformation field.  Each
figure has one row per beta and one column for every one of the `N+1` CP(N)
channels.  One-dimensional fields are rendered as narrow heatmaps; 3D fields
show the 2D plane through the source on the final lattice axis.  Field figures
are saved alongside the training data unless `--no-save-plot` is supplied.

For example:

```shell
python cpn_OPF_analysis.py --N 2 --action Gauge --dims 2 --Ls 64 --betas 4 --run
```

The saved artifact and variance-ratio names include `N`, dimension, lattice
size, beta, action type, and the selected `i`, `j` OPF indices, so different
analyses do not overwrite each other.
