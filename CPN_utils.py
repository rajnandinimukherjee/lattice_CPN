import os
import pdb
import time
import torch


def CNtoR2N(x, matrix=False):
    if matrix:
        assert x.shape[-1] == x.shape[-2], \
            "input must be square matrix in last 2 dims"
        new_vec = torch.zeros(
            x.shape[:-2]+(2*x.shape[-2], 2*x.shape[-1]),
            dtype=x.dtype)
        new_vec[..., 0::2, 0::2] = x.real
        new_vec[..., 0::2, 1::2] = -x.imag
        new_vec[..., 1::2, 0::2] = x.imag
        new_vec[..., 1::2, 1::2] = x.real
    else:
        new_vec = torch.zeros(
            x.shape[:-1] + (2 * x.shape[-1],), dtype=torch.double)
        new_vec[..., 0::2], new_vec[..., 1::2] = x.real, x.imag
    return new_vec


def R2NtoCN(X):
    return X[..., 0::2] + 1j * X[..., 1::2]


def randSN(N, size=()):
    """ generates a random variable from S^N set
        in form of a normalised R^{N+1} vector """

    SN = torch.randn(size=size+(N+1,))
    return SN/torch.linalg.norm(SN, dim=-1).unsqueeze(-1)


def randCPN(N, size=()):
    """ generates a random CP(N) vector """

    return R2NtoCN(randSN(2*N+1, size=size))


def oneCPN(N, size=()):
    """ generates the CP(N) vector (1,0,...,0) """
    base_vec = torch.zeros(N+1, dtype=torch.complex128)
    base_vec[0] = 1.0
    return base_vec.repeat(*size, 1)


def small_randSUN(N, size=(), eps=0.1):
    # Random Hermitian matrix
    X = torch.randn(size=size+(N, N,), dtype=torch.cdouble)
    H = (X + X.conj().mT) / 2.0
    # Make traceless
    tr = H.diagonal(dim1=-2, dim2=-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(N, dtype=H.dtype).expand(*size, -1, -1)
    H_no_trace = H - (tr/N)*I
    # Exponentiate: U = exp(i ε H)
    U = torch.matrix_exp(1j * eps * H)
    return U


def randSUN(N, size=(), dtype=torch.complex128):
    """ generates random elements of the SU(N)
    group in NxN matrix representation """

    U = torch.randn(size=size+(N, N,)) +\
        1j*torch.randn(size=size+(N, N,))
    U = U.to(dtype)
    q, r = torch.linalg.qr(U)

    # 3) Fix phases so diag(r) has positive real entries
    diag = torch.diagonal(r, dim1=-2, dim2=-1)
    phases = diag / diag.abs()
    eye = torch.eye(N).expand(*size, N, N).clone().to(dtype)
    eye[..., torch.arange(N), torch.arange(N)] = phases.conj()
    q = q @ eye

    # 4) Project to SU(N): fix determinant
    det_q = torch.linalg.det(q)
    fix = torch.eye(N).expand(*size, N, N).clone().to(dtype)
    fix[..., -1, -1] = det_q.conj()
    q = q @ fix

    return q

    Q, R = torch.linalg.qr(U, mode='complete')

    # enforce unitarity by making diag elements of R have norm 1
    D = torch.exp(-1j*torch.angle(torch.einsum('...ii->...i', R))
                  )[..., None, :]
    U = Q*D

    # make determinant 1
    detU = torch.linalg.det(U)
    U /= detU[..., None, None]**(1/N)
    return U


def Lambda(Y):
    # return (1 + torch.linalg.norm(Y, axis=-1)**2)**0.5
    return (1+torch.einsum('...i,...i->...', Y.conj(), Y))**0.5


def call_PDF(filename, show=True):
    """ plots matplotlib graphics to pdfs, saves and opens file"""

    pdf = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pdf, format="pdf")
    pdf.close()
    plt.close("all")
    if show:
        os.system("open " + filename)


def compCat(x):
    """ concatenates real and imaginary parts 
    of a complex tensor along the last dimension """
    if x.dtype != torch.cdouble:
        x = x.to(torch.cdouble)
    return torch.cat([x.real, x.imag], dim=-1)


def compUncat(x):
    assert x.shape[-1] % 2 == 0, "last dimension size must be even"
    dim = int(x.shape[-1]/2)
    return x[..., :dim]+1j*x[..., dim:]
