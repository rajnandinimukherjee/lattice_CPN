"""XY-model one-point-function deformation study.

Modules are grouped by responsibility:

* :mod:`xy_opf.persistence` manages result files;
* :mod:`xy_opf.training` provides shared neural-training infrastructure;
* :mod:`xy_opf.objectives` implements the learned objectives;
* :mod:`xy_opf.optimization` contains numerical solvers;
* :mod:`xy_opf.plotting` produces scaling plots; and
* :mod:`xy_opf.cli` and :mod:`xy_opf.study` provide the entry points.
"""

from .config import LOSS_FUNCTIONS

__all__ = ["LOSS_FUNCTIONS"]
