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
