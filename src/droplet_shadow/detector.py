"""Event-level detector response; shot noise is sampled once per exposure."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import SimulationConfig
from .constants import FWHM_SIGMA


def pixel_grid(config: SimulationConfig) -> np.ndarray:
    width_m = config.run.raster_width_mm * 1e-3
    pixel_m = config.detector.pixel_um * 1e-6
    n = int(np.ceil(width_m / pixel_m))
    if n % 2:
        n += 1
    return (np.arange(n + 1) - n / 2) * pixel_m


def make_images(config: SimulationConfig, hits: dict[str, np.ndarray],
                rng: np.random.Generator) -> dict[str, np.ndarray]:
    edges = pixel_grid(config)
    detector = config.detector
    x, y = hits["x_m"], hits["y_m"]
    eligible = (~hits["blocked"]) & (x*x + y*y <= (detector.diameter_mm * 5e-4) ** 2)
    x, y = x[eligible], y[eligible]
    n_sim = config.run.n_simulated
    weight = config.run.exposure_electrons / n_sim
    valid_weight = np.full(len(x), weight)
    impulse, _, _ = np.histogram2d(y, x, bins=(edges, edges), weights=valid_weight)
    sigma_pixel = detector.psf_fwhm_um / FWHM_SIGMA / detector.pixel_um
    expected = gaussian_filter(impulse * detector.efficiency * detector.gain_mean,
                               sigma_pixel, mode="constant")
    # One compound-Poisson detector draw from the transport-estimated impulse
    # intensity. Sampling each weighted MC ray repeatedly at the exact same
    # coordinate would add artificial event-position clustering when weight>1.
    detected = rng.poisson(impulse * detector.efficiency)
    noisy_impulse = np.zeros_like(impulse)
    active = detected > 0
    noisy_impulse[active] = rng.gamma(detected[active] * detector.gain_shape,
                                      detector.gain_mean / detector.gain_shape)
    observed = gaussian_filter(noisy_impulse, sigma_pixel, mode="constant")
    observed += rng.poisson(detector.background_counts_pixel, size=observed.shape)
    if detector.read_noise_rms:
        observed += rng.normal(0, detector.read_noise_rms, size=observed.shape)
    return {"ideal": impulse, "expected": expected, "observed": observed,
            "x_edges_m": edges, "y_edges_m": edges,
            "detected_fraction": np.array(len(x) / n_sim)}
