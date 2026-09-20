from __future__ import annotations

import torch


def SUN_generators(N: int, cartan: bool = False) -> torch.Tensor:
    """Returns the generators of the SU(N) Lie algebra."""
    if N < 1:
        raise ValueError("N must be positive")
    generators = []

    # Off-diagonal Hermitian matrices
    if not cartan:
        for i in range(N):
            for j in range(i + 1, N):
                mat = torch.zeros((N, N), dtype=torch.complex128)
                mat[i, j] = 1
                mat[j, i] = 1
                generators.append(mat)

                mat = torch.zeros((N, N), dtype=torch.complex128)
                mat[i, j] = -1j
                mat[j, i] = 1j
                generators.append(mat)

    # Diagonal traceless Hermitian matrices
    for k in range(1, N):
        mat = torch.zeros((N, N), dtype=torch.complex128)
        for i in range(k):
            mat[i, i] = 1
        mat[k, k] = -k
        mat /= (k * (k + 1) / 2) ** 0.5
        generators.append(mat)

    if not generators:
        return torch.empty((0, N, N), dtype=torch.complex128)
    return torch.stack(generators) / 2
