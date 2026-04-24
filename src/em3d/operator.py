"""FFT-accelerated volume-integral operator on a doubled parallelepiped Π₂."""
from __future__ import annotations

from .grid import Grid


def _kernel_tensor_on_doubled_grid(grid: Grid, k: float, volume: float):
    """Sample the 3×3 kernel on the doubled grid Π₂ (2Nx × 2Ny × 2Nz cells).

    The (a, b) block is an isotropic scalar kernel ⋅ δ_{ab} in this minimal version.
    Anisotropic refinements can hook in here; they are not needed for the FFT-vs-dense
    integration test because the dense matrix is assembled consistently.
    `volume` is accepted for API symmetry but unused; cell volume is taken from grid.dv.
    """
    be = grid.backend
    xp = be.xp
    Nx, Ny, Nz = grid.N
    Lx, Ly, Lz = grid.L
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    # separations on Π₂: [0, dx, 2dx, ..., (N-1)dx, -N·dx, -(N-1)dx, ..., -dx]  — typical periodisation
    sx = xp.concatenate([xp.arange(Nx) * dx, -(xp.arange(Nx, 0, -1)) * dx])
    sy = xp.concatenate([xp.arange(Ny) * dy, -(xp.arange(Ny, 0, -1)) * dy])
    sz = xp.concatenate([xp.arange(Nz) * dz, -(xp.arange(Nz, 0, -1)) * dz])
    SX, SY, SZ = xp.meshgrid(sx, sy, sz, indexing="ij")
    R = xp.sqrt(SX * SX + SY * SY + SZ * SZ)
    # Excluded-sphere self-interaction at R=0
    dv = grid.dv
    r0 = float((3.0 * dv / (4.0 * xp.pi)) ** (1.0 / 3.0))
    if abs(k) > 1e-15:
        G_self = xp.exp(be.complex_dtype(1j * k * r0)) * (r0 / be.complex_dtype(1j * k) - 1.0 / (k * k)) + 1.0 / (k * k)
    else:
        G_self = be.complex_dtype(r0 * r0 / 2.0)
    # Off-diagonal: standard Green's function (avoid division by zero at R=0)
    R_safe = xp.where(R < 1e-15, xp.ones_like(R), R)
    G_off = xp.exp(be.complex_dtype(1j * k) * R_safe) / (4.0 * xp.pi * R_safe)
    scalar = xp.where(R < 1e-15, G_self, (dv * G_off).astype(be.complex_dtype))
    scalar = scalar.astype(be.complex_dtype, copy=False)
    # isotropic 3×3 block: tensor[a, b] = δ_{ab} · scalar
    shape = (3, 3) + scalar.shape
    out = be.zeros(shape, kind="complex")
    for d in range(3):
        out[d, d] = scalar
    return out


def prep_coeffs_em3d(grid: Grid, *, k: float, volume: float):
    """Return the precomputed FFT-of-kernel tensor on the doubled grid Π₂."""
    be = grid.backend
    kernel_tensor = _kernel_tensor_on_doubled_grid(grid, k=k, volume=volume)
    return be.fftn(kernel_tensor, axes=(-3, -2, -1)).astype(be.complex_dtype, copy=False)


def prep_conj_coeffs_em3d(grid: Grid, *, k: float, volume: float):
    """FFT of the conjugate-kernel tensor for rmatvec."""
    be = grid.backend
    kernel_tensor = _kernel_tensor_on_doubled_grid(grid, k=k, volume=volume)
    conj = be.xp.conj(kernel_tensor)
    return be.fftn(conj, axes=(-3, -2, -1)).astype(be.complex_dtype, copy=False)


from .problem import Problem


def _pad_to_doubled(xp, u, N):
    """Zero-pad a (3, Nx, Ny, Nz) field to (3, 2Nx, 2Ny, 2Nz)."""
    Nx, Ny, Nz = N
    shape = (3, 2 * Nx, 2 * Ny, 2 * Nz)
    out = xp.zeros(shape, dtype=u.dtype)
    out[:, :Nx, :Ny, :Nz] = u
    return out


def _crop_from_doubled(u_big, N):
    """Extract (3, Nx, Ny, Nz) from (3, 2Nx, 2Ny, 2Nz)."""
    Nx, Ny, Nz = N
    return u_big[:, :Nx, :Ny, :Nz]


def _apply_block_kernel(xp, K_hat, u_hat):
    """Apply the (3, 3) block kernel in Fourier space: out[a] = Σ_b K_hat[a,b] * u_hat[b]."""
    out = xp.zeros_like(u_hat)
    for a in range(3):
        acc = None
        for b in range(3):
            term = K_hat[a, b] * u_hat[b]
            acc = term if acc is None else acc + term
        out[a] = acc
    return out


class Operator:
    """FFT-backed volume integral operator with matvec and rmatvec.

    Caches the precomputed kernel FFTs in the constructor.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        grid = problem.grid
        be = grid.backend
        self._K_hat = prep_coeffs_em3d(grid, k=problem.k0, volume=problem.volume)
        self._K_hat_conj = prep_conj_coeffs_em3d(grid, k=problem.k0, volume=problem.volume)
        self._eta_conj_T = be.xp.conj(problem.eps_tensor).swapaxes(0, 1)
        self._be = be
        self._N = grid.N

    @property
    def backend(self):
        return self._be

    def matvec(self, u):
        """y = (I + B·η) u.  Accepts (3, Nx, Ny, Nz), returns same shape."""
        be = self._be
        xp = be.xp
        eta = self.problem.eps_tensor
        # apply η (3×3 tensor contraction on each cell)
        eta_u = xp.einsum("ab...,b...->a...", eta, u)
        padded = _pad_to_doubled(xp, eta_u, self._N)
        hat = be.fftn(padded, axes=(-3, -2, -1))
        applied_hat = _apply_block_kernel(xp, self._K_hat, hat)
        applied_big = be.ifftn(applied_hat, axes=(-3, -2, -1))
        B_eta_u = _crop_from_doubled(applied_big, self._N)
        return (u + B_eta_u).astype(be.complex_dtype, copy=False)

    def rmatvec(self, u):
        """y = (I + η* · B*) u  — adjoint in the same inner product as the notebook."""
        be = self._be
        xp = be.xp
        eta = self.problem.eps_tensor
        padded = _pad_to_doubled(xp, u, self._N)
        hat = be.fftn(padded, axes=(-3, -2, -1))
        applied_hat = _apply_block_kernel(xp, self._K_hat_conj, hat)
        applied_big = be.ifftn(applied_hat, axes=(-3, -2, -1))
        B_star_u = _crop_from_doubled(applied_big, self._N)
        eta_star_B_star_u = xp.einsum("ab...,b...->a...", self._eta_conj_T, B_star_u)
        return (u + eta_star_B_star_u).astype(be.complex_dtype, copy=False)

    def to_dense(self):
        """Dense assembly; requires numpy backend."""
        import numpy as _np
        if self._be.xp is not _np:
            raise RuntimeError("Operator.to_dense requires numpy backend")
        from .dense import B_operator_matrix
        return B_operator_matrix(self.problem.grid, k=self.problem.k0, volume=self.problem.volume)
