from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import numpy as np

from .grid import Grid
from .layout import flatten_field, unflatten_field

AdjointStorage = Literal["explicit", "derived", "none"]
KernelBuildStrategy = Literal["standard", "streamed"]


def _doubled_offsets(grid: Grid):
    be = grid.backend
    xp = be.xp
    nx, ny, nz = grid.N
    lx, ly, lz = grid.L
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    sx = xp.concatenate([xp.arange(nx) * dx, -(xp.arange(nx, 0, -1)) * dx])
    sy = xp.concatenate([xp.arange(ny) * dy, -(xp.arange(ny, 0, -1)) * dy])
    sz = xp.concatenate([xp.arange(nz) * dz, -(xp.arange(nz, 0, -1)) * dz])
    return sx, sy, sz


def _kernel_geometry(grid: Grid, *, k: float):
    """Build reusable scalar geometry for one streamed kernel construction."""

    be = grid.backend
    xp = be.xp
    sx, sy, sz = _doubled_offsets(grid)
    r2 = sx[:, None, None] ** 2
    r2 = r2 + sy[None, :, None] ** 2
    r2 = r2 + sz[None, None, :] ** 2
    radius = xp.sqrt(r2)
    is_self = radius < 1e-15
    radius = xp.where(is_self, xp.ones_like(radius), radius)
    inv_radius = 1.0 / radius
    ik = be.complex_dtype(1j * k)
    weighted_green = (
        xp.exp(ik * radius) / (4.0 * xp.pi * radius) * grid.dv
    ).astype(be.complex_dtype, copy=False)
    coefficient_1 = (
        3.0 * inv_radius * inv_radius - 3.0 * ik * inv_radius - k * k
    ).astype(be.complex_dtype, copy=False)
    coefficient_2 = (
        k * k + ik * inv_radius - inv_radius * inv_radius
    ).astype(be.complex_dtype, copy=False)
    return {
        "offsets": (sx, sy, sz),
        "inv_radius": inv_radius,
        "is_self": is_self,
        "weighted_green": weighted_green,
        "coefficient_1": coefficient_1,
        "coefficient_2": coefficient_2,
    }


def _kernel_block_from_geometry(grid: Grid, geometry, row: int, column: int):
    be = grid.backend
    xp = be.xp
    offsets = geometry["offsets"]
    inv_radius = geometry["inv_radius"]
    coordinate_shapes = (
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
    )
    alpha_row = offsets[row].reshape(coordinate_shapes[row]) * inv_radius
    alpha_column = offsets[column].reshape(coordinate_shapes[column]) * inv_radius
    block = (
        geometry["weighted_green"]
        * geometry["coefficient_1"]
        * alpha_row
        * alpha_column
    )
    if row == column:
        block = block + geometry["weighted_green"] * geometry["coefficient_2"]
        block = xp.where(geometry["is_self"], -1.0 / 3.0, block)
    else:
        block = xp.where(geometry["is_self"], 0.0, block)
    return block.astype(be.complex_dtype, copy=False)


def _kernel_tensor_on_doubled_grid(
    grid: Grid,
    k: float,
    volume: float | None = None,
):
    """Build the block-circulant embedding of the dyadic kernel.

    ``volume`` is retained for backward compatibility. The integration weight
    is the cell volume ``grid.dv``; the total domain volume is not used.
    """

    be = grid.backend
    geometry = _kernel_geometry(grid, k=k)
    doubled_shape = tuple(2 * value for value in grid.N)
    tensor = be.empty((3, 3) + doubled_shape, kind="complex")
    for row in range(3):
        for column in range(3):
            tensor[row, column] = _kernel_block_from_geometry(
                grid, geometry, row, column
            )
    return tensor


def _build_kernel_hat_streamed(
    grid: Grid,
    *,
    k: float,
    explicit_adjoint: bool,
):
    """Build FFT blocks one at a time instead of materialising all spatial blocks."""

    be = grid.backend
    geometry = _kernel_geometry(grid, k=k)
    doubled_shape = tuple(2 * value for value in grid.N)
    kernel_hat = be.empty((3, 3) + doubled_shape, kind="complex")
    kernel_hat_adjoint = (
        be.empty((3, 3) + doubled_shape, kind="complex")
        if explicit_adjoint
        else None
    )
    for row in range(3):
        for column in range(3):
            block = _kernel_block_from_geometry(grid, geometry, row, column)
            kernel_hat[row, column] = be.fftn(
                block, axes=(-3, -2, -1)
            ).astype(be.complex_dtype, copy=False)
            if kernel_hat_adjoint is not None:
                kernel_hat_adjoint[row, column] = be.fftn(
                    be.xp.conj(block), axes=(-3, -2, -1)
                ).astype(be.complex_dtype, copy=False)
            del block
    return kernel_hat, kernel_hat_adjoint


def _reverse_frequency_axes(xp, block):
    """Map ``F(k)`` to ``F(-k)`` for NumPy/CuPy FFT index ordering."""

    return xp.roll(
        xp.flip(block, axis=(-3, -2, -1)),
        shift=(1, 1, 1),
        axis=(-3, -2, -1),
    )


@dataclass(frozen=True)
class PreparedEMKernel:
    """Reusable FFT representation of the electrodynamic convolution kernel.

    ``adjoint_storage='derived'`` stores only the forward spectral tensor and
    reconstructs each adjoint block transiently from the exact DFT identity
    ``FFT(conj(K))[k] = conj(FFT(K)[-k])``. This avoids a second persistent
    nine-block tensor, which is decisive on large three-dimensional grids.
    """

    grid_shape: tuple[int, int, int]
    grid_lengths: tuple[float, float, float]
    wave_number: float
    device: str
    precision: str
    kernel_hat: object
    kernel_hat_adjoint: object | None
    adjoint_storage: AdjointStorage = "explicit"
    build_strategy: KernelBuildStrategy = "standard"

    @classmethod
    def build(
        cls,
        grid: Grid,
        *,
        k: float,
        include_adjoint: bool = True,
        adjoint_storage: Literal["explicit", "derived"] = "explicit",
        build_strategy: KernelBuildStrategy = "standard",
    ) -> "PreparedEMKernel":
        if build_strategy not in {"standard", "streamed"}:
            raise ValueError(
                "build_strategy must be 'standard' or 'streamed', "
                f"got {build_strategy!r}"
            )
        if adjoint_storage not in {"explicit", "derived"}:
            raise ValueError(
                "adjoint_storage must be 'explicit' or 'derived', "
                f"got {adjoint_storage!r}"
            )
        storage: AdjointStorage = (
            adjoint_storage if include_adjoint else "none"
        )
        explicit_adjoint = storage == "explicit"
        be = grid.backend
        if build_strategy == "streamed":
            kernel_hat, kernel_hat_adjoint = _build_kernel_hat_streamed(
                grid,
                k=k,
                explicit_adjoint=explicit_adjoint,
            )
        else:
            tensor = _kernel_tensor_on_doubled_grid(grid, k=k)
            kernel_hat = be.fftn(tensor, axes=(-3, -2, -1)).astype(
                be.complex_dtype, copy=False
            )
            kernel_hat_adjoint = None
            if explicit_adjoint:
                kernel_hat_adjoint = be.fftn(
                    be.xp.conj(tensor), axes=(-3, -2, -1)
                ).astype(be.complex_dtype, copy=False)
            del tensor
        return cls(
            grid_shape=tuple(int(value) for value in grid.N),
            grid_lengths=tuple(float(value) for value in grid.L),
            wave_number=float(k),
            device=str(be.device),
            precision=str(be.precision.value),
            kernel_hat=kernel_hat,
            kernel_hat_adjoint=kernel_hat_adjoint,
            adjoint_storage=storage,
            build_strategy=build_strategy,
        )

    @property
    def supports_adjoint(self) -> bool:
        return self.adjoint_storage != "none"

    @property
    def persistent_nbytes(self) -> int:
        total = int(getattr(self.kernel_hat, "nbytes", 0))
        if self.kernel_hat_adjoint is not None:
            total += int(getattr(self.kernel_hat_adjoint, "nbytes", 0))
        return total

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
    return PreparedEMKernel.build(
        grid, k=k, include_adjoint=False
    ).kernel_hat


def prep_conj_coeffs_em3d(
    grid: Grid,
    *,
    k: float,
    volume: float | None = None,
):
    return PreparedEMKernel.build(
        grid,
        k=k,
        include_adjoint=True,
        adjoint_storage="explicit",
    ).kernel_hat_adjoint


from .problem import Problem


def _pad_to_doubled(xp, field, shape):
    nx, ny, nz = shape
    result = xp.zeros((3, 2 * nx, 2 * ny, 2 * nz), dtype=field.dtype)
    result[:, :nx, :ny, :nz] = field
    return result


def _crop_from_doubled(field, shape):
    nx, ny, nz = shape
    return field[:, :nx, :ny, :nz]


def _apply_block_kernel(xp, kernel_hat, field_hat):
    result = xp.zeros_like(field_hat)
    for row in range(3):
        accumulator = None
        for column in range(3):
            term = kernel_hat[row, column] * field_hat[column]
            accumulator = term if accumulator is None else accumulator + term
        result[row] = accumulator
    return result


def _apply_block_kernel_adjoint_derived(xp, kernel_hat, field_hat):
    result = xp.zeros_like(field_hat)
    for row in range(3):
        accumulator = None
        for column in range(3):
            reversed_block = _reverse_frequency_axes(
                xp, kernel_hat[row, column]
            )
            term = xp.conj(reversed_block) * field_hat[column]
            accumulator = term if accumulator is None else accumulator + term
            del reversed_block
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
        if prepared_kernel is None:
            prepared_kernel = PreparedEMKernel.build(grid, k=problem.k0)
        else:
            prepared_kernel.validate(grid, k=problem.k0)
        self._prepared_kernel = prepared_kernel
        self._K_hat = prepared_kernel.kernel_hat
        self._K_hat_conj = prepared_kernel.kernel_hat_adjoint
        self._eta_conj_T = None
        self._be = grid.backend
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
        prepared = self._prepared_kernel
        if not prepared.supports_adjoint:
            raise RuntimeError(
                "adjoint action is unavailable: build PreparedEMKernel with "
                "include_adjoint=True"
            )
        be = self._be
        xp = be.xp
        padded = _pad_to_doubled(xp, field, self._N)
        transformed = be.fftn(padded, axes=(-3, -2, -1))
        if prepared.adjoint_storage == "explicit":
            assert self._K_hat_conj is not None
            applied_hat = _apply_block_kernel(
                xp, self._K_hat_conj, transformed
            )
        elif prepared.adjoint_storage == "derived":
            applied_hat = _apply_block_kernel_adjoint_derived(
                xp, self._K_hat, transformed
            )
        else:  # defensive; ``supports_adjoint`` handled ``none`` above
            raise RuntimeError(
                f"unsupported adjoint storage {prepared.adjoint_storage!r}"
            )
        applied_big = be.ifftn(applied_hat, axes=(-3, -2, -1))
        kernel_adjoint_field = _crop_from_doubled(applied_big, self._N)
        if self._eta_conj_T is None:
            self._eta_conj_T = xp.conj(self.problem.eps_tensor).swapaxes(0, 1)
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
        """Backward-compatible alias returning the kernel matrix ``B``."""

        return self.to_dense_kernel()
