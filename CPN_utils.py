from __future__ import annotations

from collections.abc import Sequence

import torch


def CNtoR2N(x: torch.Tensor, matrix: bool = False) -> torch.Tensor:
    """Convert a complex vector or matrix to its real representation."""
    if matrix:
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("input must be square in its final two dimensions")
        real_dtype = x.real.dtype
        new_matrix = torch.zeros(
            x.shape[:-2] + (2 * x.shape[-2], 2 * x.shape[-1]),
            dtype=real_dtype,
            device=x.device,
        )
        new_matrix[..., 0::2, 0::2] = x.real
        new_matrix[..., 0::2, 1::2] = -x.imag
        new_matrix[..., 1::2, 0::2] = x.imag
        new_matrix[..., 1::2, 1::2] = x.real
        return new_matrix

    return torch.view_as_real(x).flatten(start_dim=-2)


def R2NtoCN(x: torch.Tensor) -> torch.Tensor:
    """Convert interleaved real and imaginary components to a complex tensor."""
    if x.shape[-1] % 2:
        raise ValueError("the final dimension must have even length")
    return torch.complex(x[..., 0::2], x[..., 1::2])


def randSN(n: int, size: Sequence[int] = ()) -> torch.Tensor:
    """Draw a uniformly distributed point on the unit sphere S^n."""
    sample = torch.randn(size=tuple(size) + (n + 1,))
    return sample / torch.linalg.vector_norm(sample, dim=-1, keepdim=True)


def randCPN(n: int, size: Sequence[int] = ()) -> torch.Tensor:
    """Draw a random normalized representative of CP(n)."""
    return R2NtoCN(randSN(2 * n + 1, size=size))


def oneCPN(n: int, size: Sequence[int] = ()) -> torch.Tensor:
    """Return copies of the CP(n) representative ``(1, 0, ..., 0)``."""
    base_vector = torch.zeros(n + 1, dtype=torch.complex128)
    base_vector[0] = 1.0
    return base_vector.repeat(*tuple(size), 1)


def small_randSUN(
    n: int,
    size: Sequence[int] = (),
    eps: float = 0.1,
) -> torch.Tensor:
    """Draw SU(n) matrices close to the identity."""
    batch_shape = tuple(size)
    x = torch.randn(size=batch_shape + (n, n), dtype=torch.cdouble)
    hermitian = (x + x.conj().mT) / 2.0
    trace = hermitian.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    identity = torch.eye(n, dtype=hermitian.dtype).expand(*batch_shape, -1, -1)
    traceless = hermitian - (trace / n) * identity
    return torch.matrix_exp(1j * eps * traceless)


def randSUN(
    n: int,
    size: Sequence[int] = (),
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Draw Haar-distributed matrices in the fundamental representation of SU(n)."""
    batch_shape = tuple(size)
    matrix = torch.randn(size=batch_shape + (n, n))
    matrix = (matrix + 1j * torch.randn_like(matrix)).to(dtype)
    q, r = torch.linalg.qr(matrix)

    diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
    phases = diagonal / diagonal.abs()
    identity = torch.eye(n, dtype=dtype).expand(*batch_shape, n, n).clone()
    indices = torch.arange(n)
    identity[..., indices, indices] = phases.conj()
    q = q @ identity

    determinant = torch.linalg.det(q)
    determinant_fix = torch.eye(n, dtype=dtype).expand(*batch_shape, n, n).clone()
    determinant_fix[..., -1, -1] = determinant.conj()
    return q @ determinant_fix


def Lambda(y: torch.Tensor) -> torch.Tensor:
    """Return the normalization factor for a homogeneous deformation."""
    return (1 + torch.einsum("...i,...i->...", y.conj(), y)) ** 0.5


def compCat(x: torch.Tensor) -> torch.Tensor:
    """Concatenate the real and imaginary parts along the final dimension."""
    if not x.is_complex():
        x = x.to(torch.cdouble)
    return torch.cat([x.real, x.imag], dim=-1)


def compUncat(x: torch.Tensor) -> torch.Tensor:
    """Undo :func:`compCat` and return a complex tensor."""
    if x.shape[-1] % 2:
        raise ValueError("the final dimension must have even length")
    midpoint = x.shape[-1] // 2
    return torch.complex(x[..., :midpoint], x[..., midpoint:])
