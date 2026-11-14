# XY OPF analysis layout

`xy_OPF_analysis.py` remains the command users run. Its implementation is
split into modules with one responsibility each:

- `cli.py`: argument definitions and validation;
- `study.py`: load, run, and delete orchestration;
- `objectives.py`: variance- and penalty-loss training workflows;
- `optimization.py`: field-scale and saddle-point solvers;
- `training.py`: shared splits, optimizer setup, and training serialization;
- `persistence.py`: artifact paths, indexes, discovery, loading, and deletion;
- `plotting.py`: cross-dimension summary plots; and
- `config.py`: shared constants and labels.

New code should import from the module that owns the operation, for example:

```python
from xy_opf.persistence import load_variance_ratios
from xy_opf.optimization import solve_saddle_point_field
```

The existing command is unchanged:

```shell
python xy_OPF_analysis.py --dims 1 2 3 --Ls 64 32 16 --no-run
```

`optimization.py` also provides `solve_lattice_laplace_field` and
`fit_lattice_laplace_profile` for Approximation 3.  The latter fixes the
source value to one, solves the masked lattice Laplace equation, and
optimizes the overall scale against the XY variance.  The radial comparison
script uses that method alongside the saved Approximation 1 and 2 profiles:

```shell
python plot_xy_radial_profiles.py --betas 0.65 0.80 1.20 1.35 --lattice-size 32
```

It expects matching XY ensembles in `data/` and writes PDF and PNG copies
under `paper/figs/`.

Approximation 3 can also be run over every ensemble matching the requested
dimensions and lattice sizes. Each beta result is saved immediately as a
`loss-approx3_result.npy` artifact, and its variance ratio is upserted into the
dimension-level index. The one dimensional ensembles use mask radius 20;
dimensions 2 and 3 use radius 6. Set `MPLBACKEND=Agg` for batch plotting:

```shell
MPLBACKEND=Agg python xy_OPF_analysis.py --run --dims 1 --Ls 64 \
  --loss-function approx3 --mask-radius 20 --no-plot-training
MPLBACKEND=Agg python xy_OPF_analysis.py --run --dims 2 3 --Ls 32 16 \
  --loss-function approx3 --mask-radius 6 --no-plot-training
MPLBACKEND=Agg python xy_OPF_analysis.py --no-run --dims 1 2 3 --Ls 64 32 16 \
  --save-plot paper/figs/fig4_xy_scaling.pdf
```

For a configuration-dependent XY deformation, select
`--loss-function field-dependent`.  The learned coefficient fields define
`F_x(theta) = a_x + b_x sin(theta_x) + c_x cos(theta_x)`, with the same
periodic support mask as the fixed-field training.  This pointwise map has a
diagonal Jacobian, whose exact determinant is included in the deformed
observable.  By default, training starts from a zero profile with a
zero-initialized U-Net output.  To warm-start from a saved field-independent
variance profile instead, pass `--field-dependent-init fixed-profile`.  For
example:

```shell
python xy_OPF_analysis.py --run --dims 2 --Ls 32 --betas 0.65 \
  --loss-function field-dependent --field-dependent-init zero \
  --mask-radius 6 --epochs 6000 --batch-size 256 \
  --learning-rate 0.001 --lr-scheduler plateau --no-plot-training
```

For the beta=0.65, 32x32 ensemble, the saved zero-start run stopped by early
stopping after 3,311 epochs. Its best validation checkpoint was at epoch
2,261 and gave a full-ensemble variance ratio of 0.1263.
