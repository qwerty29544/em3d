from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from .grid import Grid
from .layout import flatten_field, unflatten_field


def _kernel_tensor_on_doubled_grid(
    grid: Grid,
    k: float,
    volume: float | None = None,
):
    """Build the block-circulant embedding of the dyadic kernel.

    ``volume`` is retained for backward compatibility.  The discretization
    weight is the cell volume ``grid.dv``; the total domain volume does not
    enter the kernel formula.
    """

    be = grid.backend
    xp = be.xp
    Nx, Ny, Nz = grid.N
    Lx, Ly, Lz = grid.L
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

    sx = xp.concatenate([xp.arange(Nx) * dx, -(xp.arange(Nx, 0, -1)) * dx])
    sy = xp.concatenate([xp.arange(Ny) * dy, -(xp.arange(Ny, 0, -1)) * dy])
    sz = xp.concatenate([xp.arange(Nz) * dz, -(xp.arange(Nz, 0, -1)) * dz])
    SX, SY, SZ = xp.meshgrid(sx, sy, sz, indexing="ij")
    R = xp.sqrt(SX * SX + SY * SY + SZ * SZ)
    is_self = R < 1e-15
    R_safe = xp.where(is_self, xp.ones_like(R), R)
    inv_R = 1.0 / R_safe
    inv_R2 = inv_R * inv_R
    ik = be.complex_dtype(1j * k)
    coef_1 = (3.0 * inv_R2) - (3.0 * ik * inv_R) - (k * k)
    coef_2 = (k * k) + (ik * inv_R) - inv_R2
    green = xp.exp(ik * R_safe) / (4.0 * xp.pi * R_safe)
    alpha = (SX / R_safe, SY / R_safe, SZ / R_safe)

    shape = (3, 3) + R.shape
    result = be.zeros(shape, kind="complex")
    for row in range(3):
        for column in range(3):
            value = (
                green
                * grid.dv
                * coef_1
                * alpha[row]
                * alpha[column]
            )
            if row == column:
                value = value + green * grid.dv * coef_2
                value = xp.where(is_self, -1.0 / 3.0, value)
            else:
                value = xp.where(is_self, 0.0, value)
            result[row, column] = value.astype(be.complex_dtype, copy=False)
    return result


@dataclass(frozen=True)
class PreparedEMKernel:
    """Reusable FFT representation of the electrodynamic convolution kernel."""

    grid_shape: tuple[int, int, int]
    grid_lengths: tuple[float, float, float]
    wave_number: float
    device: str
    precision: str
    kernel_hat: object
    kernel_hat_adjoint: object

    @classmethod
    def build(cls, grid: Grid, *, k: float) -> "PreparedEMKernel":
        tensor = _kernel_tensor_on_doubled_grid(grid, k=k)
        be = grid.backend
        kernel_hat = be.fftn(tensor, axes=(-3, -2, -1)).astype(
            be.complex_dtype, copy=False
        )
        kernel_hat_adjoint = be.fftn(
            be.xp.conj(tensor), axes=(-3, -2, -1)
        ).astype(be.complex_dtype, copy=False)
        return cls(
            grid_shape=tuple(int(value) for value in grid.N),
            grid_lengths=tuple(float(value) for value in grid.L),
            wave_number=float(k),
            device=str(be.device),
            precision=str(be.precision.value),
            kernel_hat=kernel_hat,
            kernel_hat_adjoint=kernel_hat_adjoint,
        )

    def validate(self, grid: Grid, *, k: float) -> None:
        errors: list[str] = []
        if tuple(grid.N) != self.grid_shape:
            errors.append(f"grid shape {grid.N!r} != {self.grid_shape!r}")
        if not np.allclose(grid.L, self.grid_lengths, rtol=0.0, atol=0.0):
            errors.append(f"grid lengths {grid.L!r} != {self.grid_lengths!r}")
        if not np.isclose(float(k), self.wave_number, rtol=1e-14, atol=1e-14):
            errors.append(f"wave number {k!r} != {self.wave_number!r}")
        if grid.backend.device != self.device:
            errors.append(f"device {grid.backend.device!r} != {self.device!r}")
        if grid.backend.precision.value != self.precision:
            errors.append(
                f"precision {grid.backend.precision.value!r} != {self.precision!r}"
            )
        if errors:
            raise ValueError("incompatible PreparedEMKernel: " + "; ".join(errors))


def prep_coeffs_em3d(
    grid: Grid,
    *,
    k: float,
    volume: float | None = None,
):
    return PreparedEMKernel.build(grid, k=k).kernel_hat


def prep_conj_coeffs_em3d(
    grid: Grid,
    *,
    k: float,
    volume: float | None = None,
):
    return PreparedEMKernel.build(grid, k=k).kernel_hat_adjoint


from .problem import Problem


def _pad_to_doubled(xp, field, shape):
    Nx, Ny, Nz = shape
    result = xp.zeros((3, 2 * Nx, 2 * Ny, 2 * Nz), dtype=field.dtype)
    result[:, :Nx, :Ny, :Nz] = field
    return result


def _crop_from_doubled(field, shape):
    Nx, Ny, Nz = shape
    return field[:, :Nx, :Ny, :Nz]


def _apply_block_kernel(xp, kernel_hat, field_hat):
    result = xp.zeros_like(field_hat)
    for row in range(3):
        accumulator = None
        for column in range(3):
            term = kernel_hat[row, column] * field_hat[column]
            accumulator = term if accumulator is None else accumulator + term
        result[row] = accumulator
    return result


class Operator:
    def __init__(
        self,
        problem: Problem,
        *,
        prepared_kernel: PreparedEMKernel | None = None,
    ):
        self.problem = problem
        grid = problem.grid
        be = grid.backend
        if prepared_kernel is None:
            prepared_kernel = PreparedEMKernel.build(grid, k=problem.k0)
        else:
            prepared_kernel.validate(grid, k=problem.k0)
        self._prepared_kernel = prepared_kernel
        self._K_hat = prepared_kernel.kernel_hat
        self._K_hat_conj = prepared_kernel.kernel_hat_adjoint
        self._eta_conj_T = be.xp.conj(problem.eps_tensor).swapaxes(0, 1)
        self._be = be
        self._N = grid.N

    @property
    def backend(self):
        return self._be

    @property
    def shape(self) -> tuple[int, int]:
        dimension = 3 * prod(self._N)
        return dimension, dimension

    @property
    def prepared_kernel(self) -> PreparedEMKernel:
        return self._prepared_kernel

    def matvec(self, field):
        be = self._be
        xp = be.xp
        contrast = self.problem.eps_tensor
        contrast_field = xp.einsum("ab...,b...->a...", contrast, field)
        padded = _pad_to_doubled(xp, contrast_field, self._N)
        transformed = be.fftn(padded, axes=(-3, -2, -1))
        applied_hat = _apply_block_kernel(xp, self._K_hat, transformed)
        applied_big = be.ifftn(applied_hat, axes=(-3, -2, -1))
        integral_term = _crop_from_doubled(applied_big, self._N)
        return (field - integral_term).astype(be.complex_dtype, copy=False)

    def rmatvec(self, field):
        be = self._be
        xp = be.xp
        padded = _pad_to_doubled(xp, field, self._N)
        transformed = be.fftn(padded, axes=(-3, -2, -1))
        applied_hat = _apply_block_kernel(xp, self._K_hat_conj, transformed)
        applied_big = be.ifftn(applied_hat, axes=(-3, -2, -1))
        kernel_adjoint_field = _crop_from_doubled(applied_big, self._N)
        contrast_adjoint_field = xp.einsum(
            "ab...,b...->a...", self._eta_conj_T, kernel_adjoint_field
        )
        return (field - contrast_adjoint_field).astype(
            be.complex_dtype, copy=False
        )

    def matvec_flat(self, vector):
        return flatten_field(self.matvec(unflatten_field(vector, self._N)))

    def rmatvec_flat(self, vector):
        return flatten_field(self.rmatvec(unflatten_field(vector, self._N)))

    def iteration_matvec_flat(self, vector, *, mu: complex):
        if abs(mu) == 0.0:
            raise ValueError("mu must be non-zero")
        return vector - self.matvec_flat(vector) / self._be.complex_dtype(mu)

    def to_dense_kernel(self) -> np.ndarray:
        if self._be.device != "cpu":
            raise RuntimeError("dense matrix construction requires a NumPy backend")
        from .dense import dense_kernel_matrix

        return dense_kernel_matrix(self.problem.grid, k=self.problem.k0)

    def to_dense_operator(self) -> np.ndarray:
        if self._be.device != "cpu":
            raise RuntimeError("dense matrix construction requires a NumPy backend")
        from .dense import dense_operator_matrix

        return dense_operator_matrix(self.problem)

    def to_dense(self) -> np.ndarray:
        """Backward-compatible alias returning the kernel matrix ``B``.

        Use :meth:`to_dense_operator` when the full matrix ``I - B chi`` is
        required.
        """

        return self.to_dense_kernel()
