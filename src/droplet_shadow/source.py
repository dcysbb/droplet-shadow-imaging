"""Finite four-dimensional crossover source and energy distribution."""

from __future__ import annotations

import numpy as np

from .config import Source
from .constants import C, E_CHARGE, FWHM_SIGMA, M_E


def phase_space_covariance(source: Source) -> np.ndarray:
    sx = source.crossover_fwhm_um * 1e-6 / FWHM_SIGMA
    sa = source.divergence_rms_mrad * 1e-3
    covariance = np.diag([sx**2, sx**2, sa**2, sa**2])
    covariance[0, 2] = covariance[2, 0] = source.x_xprime_correlation * sx * sa
    covariance[1, 3] = covariance[3, 1] = source.y_yprime_correlation * sx * sa
    return covariance


def geometric_emittance(source: Source) -> tuple[float, float]:
    cov = phase_space_covariance(source)
    return tuple(float(np.sqrt(np.linalg.det(cov[np.ix_(index, index)])))
                 for index in ([0, 2], [1, 3]))


def sample_source(source: Source, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    phase = rng.multivariate_normal(np.zeros(4), phase_space_covariance(source), size=n)
    kinetic_j = rng.normal(source.energy_mev, source.energy_mev * source.energy_spread_rms, size=n) * 1e6 * E_CHARGE
    if np.any(kinetic_j <= 0):
        raise ValueError("Energy spread generated nonpositive electron kinetic energy")
    momentum = np.sqrt(kinetic_j * (kinetic_j + 2 * M_E * C**2)) / C
    direction = np.column_stack((phase[:, 2], phase[:, 3], np.ones(n)))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return phase, momentum[:, None] * direction

