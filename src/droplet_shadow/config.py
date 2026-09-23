"""Validated, unit-explicit simulation configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Geometry:
    radius_um: float = 50.0
    l1_mm: float = 10.0
    l2_mm: float = 500.0
    temperature_k: float = 298.0
    epsilon_water: float = 78.0
    surface_tension_n_m: float = 0.072

    def __post_init__(self) -> None:
        if min(self.radius_um, self.l1_mm, self.l2_mm, self.temperature_k,
               self.epsilon_water, self.surface_tension_n_m) <= 0:
            raise ValueError("Geometry lengths, temperature, dielectric constant and tension must be positive")

    @property
    def radius_m(self) -> float:
        return self.radius_um * 1e-6

    @property
    def magnification(self) -> float:
        return (self.l1_mm + self.l2_mm) / self.l1_mm


@dataclass(frozen=True)
class Charge:
    model: str = "surface"  # surface, volume, neutral_double_layer, poisson_boltzmann
    q_e: float = 1e6
    double_layer_q_e: float = 0.0
    double_layer_thickness_nm: float = 1.0
    dipole_potential_v: float = 0.0
    dipole_thickness_nm: float = 0.5
    dipole_epsilon: float = 2.0
    ion_positive_count: float = 0.0
    ion_negative_count: float = 0.0
    surface_fixed_e: float = 0.0
    dipole_fraction: float = 0.0
    quadrupole_fraction: float = 0.0
    patch_fraction: float = 0.0
    patch_width_deg: float = 30.0
    multipole_order: int = 8

    def __post_init__(self) -> None:
        if self.model not in {"surface", "volume", "neutral_double_layer", "poisson_boltzmann"}:
            raise ValueError(f"Unknown charge model: {self.model}")
        if self.double_layer_thickness_nm <= 0 or self.dipole_thickness_nm <= 0 or self.dipole_epsilon <= 0:
            raise ValueError("Layer thicknesses and dipole-layer permittivity must be positive")
        if self.ion_positive_count < 0 or self.ion_negative_count < 0:
            raise ValueError("Ion counts must be non-negative")
        if not 0 <= self.multipole_order <= 16:
            raise ValueError("multipole_order must be 0..16")
        if self.patch_width_deg <= 0:
            raise ValueError("patch_width_deg must be positive")
        if self.model == "poisson_boltzmann" and (
            self.double_layer_q_e or self.dipole_potential_v or
            self.dipole_fraction or self.quadrupole_fraction or self.patch_fraction
        ):
            raise ValueError("PB plus another interface charge/polarization requires a coupled ion re-solve")
        if self.model != "poisson_boltzmann" and (
            self.ion_positive_count or self.ion_negative_count or self.surface_fixed_e
        ):
            raise ValueError("Ion counts and fixed PB surface charge require the poisson_boltzmann model")


@dataclass(frozen=True)
class Source:
    energy_mev: float = 3.0
    energy_spread_rms: float = 0.001
    crossover_fwhm_um: float = 10.0
    divergence_rms_mrad: float = 10.0
    x_xprime_correlation: float = 0.0
    y_yprime_correlation: float = 0.0

    def __post_init__(self) -> None:
        if self.energy_mev <= 0 or self.energy_spread_rms < 0 or self.crossover_fwhm_um <= 0 or self.divergence_rms_mrad <= 0:
            raise ValueError("Source energy and widths must be positive")
        if abs(self.x_xprime_correlation) >= 1 or abs(self.y_yprime_correlation) >= 1:
            raise ValueError("Source correlations must have magnitude below one")


@dataclass(frozen=True)
class Detector:
    psf_fwhm_um: float = 50.0
    pixel_um: float = 25.0
    diameter_mm: float = 40.0
    efficiency: float = 1.0
    gain_mean: float = 1.0
    gain_shape: float = 4.0
    background_counts_pixel: float = 0.0
    read_noise_rms: float = 0.0

    def __post_init__(self) -> None:
        if min(self.psf_fwhm_um, self.pixel_um, self.diameter_mm,
               self.gain_mean, self.gain_shape) <= 0:
            raise ValueError("Detector widths, size and gain parameters must be positive")
        if not 0 <= self.efficiency <= 1 or min(self.background_counts_pixel, self.read_noise_rms) < 0:
            raise ValueError("Invalid detector efficiency or noise")


@dataclass(frozen=True)
class Run:
    engine: str = "ray"  # ray, ideal_occluder, geant4
    n_simulated: int = 100000
    exposure_electrons: int = 100000
    seed: int = 20260923
    raster_width_mm: float = 12.0
    transport_step_um: float = 10.0
    geant4_executable: str | None = None
    droplet_material: str = "water"  # water or vacuum
    geant4_world_step_mm: float = 5.0
    geant4_far_step_mm: float = 0.25
    geant4_near_step_um: float = 10.0
    geant4_core_step_um: float = 0.5
    geant4_cut_um: float = 1.0
    geant4_layer_step_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.engine not in {"ray", "ideal_occluder", "geant4"}:
            raise ValueError("engine must be ray, ideal_occluder or geant4")
        if self.droplet_material not in {"water", "vacuum"}:
            raise ValueError("droplet_material must be water or vacuum")
        if min(self.n_simulated, self.exposure_electrons, self.raster_width_mm,
               self.transport_step_um, self.geant4_world_step_mm,
               self.geant4_far_step_mm, self.geant4_near_step_um,
               self.geant4_core_step_um, self.geant4_cut_um,
               self.geant4_layer_step_scale) <= 0:
            raise ValueError("Run counts and widths must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    geometry: Geometry = field(default_factory=Geometry)
    charge: Charge = field(default_factory=Charge)
    source: Source = field(default_factory=Source)
    detector: Detector = field(default_factory=Detector)
    run: Run = field(default_factory=Run)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **sections: dict[str, Any]) -> "SimulationConfig":
        updated = self
        for section, changes in sections.items():
            updated = replace(updated, **{section: replace(getattr(updated, section), **changes)})
        return updated


def load_config(path: str | Path | None = None) -> SimulationConfig:
    raw = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {"geometry": Geometry, "charge": Charge, "source": Source,
             "detector": Detector, "run": Run}
    unexpected = set(raw) - set(known)
    if unexpected:
        raise ValueError(f"Unknown configuration sections: {sorted(unexpected)}")
    return SimulationConfig(**{key: cls(**raw.get(key, {})) for key, cls in known.items()})
