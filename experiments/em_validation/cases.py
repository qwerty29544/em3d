from __future__ import annotations

from dataclasses import replace

import numpy as np

from em3d.geometry import Ellipsoid
from em3d.experiments.spectral_transfer import (
    SpectralCaseDefinition,
    default_article_cases,
    local_inclusion_case,
)


def stationary_case_catalog() -> dict[str, SpectralCaseDefinition]:
    article = {case.key: case for case in default_article_cases()}
    # The larger inclusion is sufficiently represented on the quick grids and
    # remains the same physical case on every discretization.
    stable = local_inclusion_case(
        0.40,
        k0=4.0,
        key_prefix="local_inclusion_stable",
        title_prefix="Локальная неоднородность: устойчивый перенос",
    )
    stress = local_inclusion_case(
        0.25,
        k0=4.0,
        key_prefix="local_inclusion_stress",
        title_prefix="Локальная неоднородность: стресс-тест переноса",
    )
    article["local_inclusion_stable"] = stable
    article["local_inclusion_stress"] = stress
    # Compatibility alias for earlier quick configurations.
    article["local_inclusion"] = replace(
        stable,
        key="local_inclusion",
        title="Область с локальной неоднородностью",
    )
    return article


def mie_sphere_case(
    *,
    eps_r: complex,
    k0a: float,
    radius: float,
    domain_length: float,
) -> SpectralCaseDefinition:
    eps = complex(eps_r)
    token_eps = f"{eps.real:g}".replace("-", "m").replace(".", "p")
    if abs(eps.imag) > 1e-15:
        token_eps += f"_i{eps.imag:g}".replace("-", "m").replace(".", "p")
    token_k = f"{float(k0a):g}".replace("-", "m").replace(".", "p")
    k0 = float(k0a) / float(radius)
    return SpectralCaseDefinition(
        key=f"mie_sphere_eps{token_eps}_k0a{token_k}",
        title=f"Сфера: eps_r={eps}, k0a={float(k0a):g}",
        domain_lengths=(float(domain_length),) * 3,
        domain_center=(0.0, 0.0, 0.0),
        k0=k0,
        background_eps_r=1.0,
        feature_eps_r=eps,
        feature_geometry=Ellipsoid(
            center=(0.0, 0.0, 0.0),
            radii=(float(radius),) * 3,
        ),
        wave_direction=(0.0, 0.0, 1.0),
        wave_amplitude=(1.0, 0.0, 0.0),
    )


__all__ = ["mie_sphere_case", "stationary_case_catalog"]
