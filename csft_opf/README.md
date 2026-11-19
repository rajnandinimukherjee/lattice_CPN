# Complex-scalar OPF analysis

`csft_OPF_analysis.py` mirrors the XY one-point-function workflow for exact
free complex-scalar ensembles.  The scaling variable is the bare lattice mass
`am`, not `beta` or `1/beta`.

The native ensemble format is the direct output of
`theory.ScalarAction.sample`: a complex NumPy array with shape
`(configurations, *lattice_shape)`.  A dictionary with keys `phi` and `am` is
also accepted.  Files are named
`data/csft_dim{d}_am{am:.3f}_L{L}_ens.npy`.

The default comparison contains direct variance minimization and
Approximations 1--3.  Every approximate normalized profile is rescaled by
minimizing the exact scalar OPF variance.

```shell
python csft_OPF_analysis.py --run --dims 1 2 3 --Ls 8 4 4 --ams 1.0 \
  --mask-radius 1 --loss-function approx3 --no-plot-training
```

The action normalization is that of the repository-level `theory.py`:
`m2=am**2` and `kappa=1/2` in the paper's notation.
