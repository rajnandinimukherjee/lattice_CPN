from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ComplexPair:
    """A complex value represented by two real tensors.

    CP(N) samples are stored as interleaved real coordinates.  A contour
    deformation makes each of those coordinates complex, which cannot be
    materialized on PyTorch's MPS backend.  ``ComplexPair`` keeps the outer
    complex value as separate real and imaginary tensors, so its arithmetic
    stays on any backend that supports the underlying real dtype.

    This is deliberately a small value type rather than a replacement for
    ``torch.Tensor``.  Use :meth:`to_complex` only at an output boundary
    (typically after moving the pair to CPU) when a native complex tensor is
    required for display or serialization.
    """

    real: torch.Tensor
    imag: torch.Tensor

    def __post_init__(self) -> None:
        if self.real.shape != self.imag.shape:
            raise ValueError("real and imaginary tensors must have the same shape")
        if self.real.device != self.imag.device:
            raise ValueError("real and imaginary tensors must be on the same device")
        if self.real.dtype != self.imag.dtype:
            raise ValueError("real and imaginary tensors must have the same dtype")
        if self.real.is_complex() or self.imag.is_complex():
            raise ValueError("ComplexPair stores real tensors only")

    @property
    def shape(self) -> torch.Size:
        """Expose the tensor shape used by field/action interfaces."""
        return self.real.shape

    def __len__(self) -> int:
        """Return the leading dimension like the wrapped field tensor."""
        return len(self.real)

    @property
    def ndim(self) -> int:
        """Expose the tensor rank used by lattice helpers."""
        return self.real.ndim

    @property
    def dtype(self) -> torch.dtype:
        """Return the real component dtype."""
        return self.real.dtype

    @property
    def device(self) -> torch.device:
        """Return the common component device."""
        return self.real.device

    def __getitem__(self, item: object) -> ComplexPair:
        return ComplexPair(self.real[item], self.imag[item])

    def __add__(self, other: ComplexPair | torch.Tensor | float) -> ComplexPair:
        if isinstance(other, ComplexPair):
            return ComplexPair(self.real + other.real, self.imag + other.imag)
        return ComplexPair(self.real + other, self.imag)

    def __radd__(self, other: torch.Tensor | float) -> ComplexPair:
        return self + other

    def __sub__(self, other: ComplexPair | torch.Tensor | float) -> ComplexPair:
        if isinstance(other, ComplexPair):
            return ComplexPair(self.real - other.real, self.imag - other.imag)
        return ComplexPair(self.real - other, self.imag)

    def __rsub__(self, other: torch.Tensor | float) -> ComplexPair:
        return ComplexPair(other - self.real, -self.imag)

    def __mul__(self, other: ComplexPair | torch.Tensor | float) -> ComplexPair:
        if isinstance(other, ComplexPair):
            return ComplexPair(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real,
            )
        return ComplexPair(self.real * other, self.imag * other)

    def __rmul__(self, other: torch.Tensor | float) -> ComplexPair:
        return self * other

    def __neg__(self) -> ComplexPair:
        return ComplexPair(-self.real, -self.imag)

    def mean(self, *args: object, **kwargs: object) -> ComplexPair:
        """Return the component-wise tensor mean."""
        return ComplexPair(self.real.mean(*args, **kwargs), self.imag.mean(*args, **kwargs))

    def sum(self, *args: object, **kwargs: object) -> ComplexPair:
        """Return the component-wise tensor sum."""
        return ComplexPair(self.real.sum(*args, **kwargs), self.imag.sum(*args, **kwargs))

    def exp(self) -> ComplexPair:
        """Return the complex exponential without a complex tensor."""
        magnitude = torch.exp(self.real)
        return ComplexPair(magnitude * torch.cos(self.imag), magnitude * torch.sin(self.imag))

    def roll(self, shifts: int | tuple[int, ...], dims: int | tuple[int, ...]) -> ComplexPair:
        """Roll both components as one lattice field."""
        return ComplexPair(
            torch.roll(self.real, shifts, dims),
            torch.roll(self.imag, shifts, dims),
        )

    def matmul(self, other: torch.Tensor) -> ComplexPair:
        """Right-multiply both components by a real matrix."""
        if other.is_complex():
            raise ValueError("ComplexPair.matmul requires a real matrix")
        return ComplexPair(self.real @ other, self.imag @ other)

    def to(self, *args: object, **kwargs: object) -> ComplexPair:
        """Move or cast both real components together."""
        return ComplexPair(self.real.to(*args, **kwargs), self.imag.to(*args, **kwargs))

    def cpu(self) -> ComplexPair:
        """Move both components to CPU for native complex output if needed."""
        return self.to("cpu")

    def to_complex(self) -> torch.Tensor:
        """Materialize a native complex tensor at an explicit output boundary."""
        return torch.complex(self.real, self.imag)


def complex_inner(
    left: torch.Tensor | ComplexPair,
    right: torch.Tensor | ComplexPair,
) -> torch.Tensor | ComplexPair:
    """Return the bilinear final-axis inner product in real-pair arithmetic."""
    if not isinstance(left, ComplexPair) and not isinstance(right, ComplexPair):
        return (left * right).sum(dim=-1)
    if not isinstance(left, ComplexPair):
        left = ComplexPair(left, torch.zeros_like(left))
    if not isinstance(right, ComplexPair):
        right = ComplexPair(right, torch.zeros_like(right))
    return (left * right).sum(dim=-1)


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
