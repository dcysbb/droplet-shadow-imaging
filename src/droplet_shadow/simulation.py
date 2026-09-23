"""Simulation orchestration, provenance and inspectable result files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

from .analysis import (angular_harmonics, caustic_locations, image_metrics,
                       radial_profile, spatial_resolution_object_um)
from .config import SimulationConfig
from .constants import C, COULOMB_K, E_CHARGE, M_E
from .detector import make_images
from .fields import build_field, multipole_convergence_error, rayleigh_charge_e
from .native import geant4_executable, run_geant4
from .rays import trace_rays
from .source import geometric_emittance, sample_source


@dataclass
class SimulationResult:
    config: SimulationConfig
    hits: dict[str, np.ndarray]
    images: dict[str, np.ndarray]
    metrics: dict[str, float | None]
    output_dir: Path | None = None


def _base_metrics(config: SimulationConfig, hits: dict[str, np.ndarray],
                  images: dict[str, np.ndarray], phase: np.ndarray,
                  momentum: np.ndarray) -> dict[str, float | None]:
    qray = rayleigh_charge_e(config.geometry.radius_m, config.geometry.surface_tension_n_m)
    radius = config.geometry.radius_m
    qe = config.charge.q_e
    primary_hits = hits["primary"]
    entered_hits = hits["entered"]
    data: dict[str, float | None] = {
        "rayleigh_charge_e": qray,
        "rayleigh_fraction": abs(qe) / qray,
        "surface_potential_from_net_charge_v": COULOMB_K * qe * E_CHARGE / radius,
        "surface_external_field_from_net_charge_v_m": COULOMB_K * qe * E_CHARGE / radius**2,
        "magnification": config.geometry.magnification,
        "resolution_fwhm_object_um": spatial_resolution_object_um(config),
        "geometric_emittance_x_m_rad": geometric_emittance(config.source)[0],
        "geometric_emittance_y_m_rad": geometric_emittance(config.source)[1],
        "primary_detector_fraction": float(primary_hits.sum() / config.run.n_simulated),
        "secondary_detector_fraction": float((~primary_hits).sum() / config.run.n_simulated),
        "entered_detector_fraction": float(entered_hits.sum() / config.run.n_simulated),
        "mean_detected_energy_mev": float(np.mean(hits["energy_mev"])) if len(hits["energy_mev"]) else None,
        "exposure_weight_per_transport_primary": (
            config.run.exposure_electrons / config.run.n_simulated),
    }
    if (config.charge.dipole_fraction or config.charge.quadrupole_fraction or
            config.charge.patch_fraction):
        data["multipole_truncation_relative_rms"] = multipole_convergence_error(config)
    if "deposited_e_by_event" in hits:
        total_deposition = float(np.sum(hits["deposited_e_by_event"]))
        data["mean_deposited_charge_e_per_primary"] = total_deposition / config.run.n_simulated
        data["projected_exposure_charge_change_e"] = (total_deposition / config.run.n_simulated *
                                                      config.run.exposure_electrons)
        data["projected_charge_change_fraction"] = (
            abs(data["projected_exposure_charge_change_e"] / config.charge.q_e)
            if config.charge.q_e else None)
    if "exit_energy_mev" in hits:
        primary_exit = hits["exit_parent_id"] == 0
        exit_events = hits["exit_event_id"][primary_exit]
        initial_p = np.linalg.norm(momentum[exit_events], axis=1)
        initial_energy = (np.sqrt(M_E**2 * C**4 + initial_p**2 * C**2) -
                          M_E * C**2) / (1e6 * E_CHARGE)
        loss = initial_energy - hits["exit_energy_mev"][primary_exit]
        data["primary_exit_fraction"] = float(primary_exit.sum() / config.run.n_simulated)
        data["mean_primary_exit_loss_kev"] = float(np.mean(loss) * 1000) if loss.size else None
        data["median_primary_exit_loss_kev"] = float(np.median(loss) * 1000) if loss.size else None
        if loss.size:
            exit_direction = np.column_stack((hits["exit_ux"][primary_exit],
                                              hits["exit_uy"][primary_exit],
                                              hits["exit_uz"][primary_exit]))
            initial_direction = momentum[exit_events] / initial_p[:, None]
            angles = np.arccos(np.clip(np.einsum("ij,ij->i", exit_direction,
                                                 initial_direction), -1, 1))
            data["primary_exit_scattering_rms_mrad"] = float(np.sqrt(np.mean(angles**2)) * 1e3)
            data["primary_exit_scattering_median_mrad"] = float(np.median(angles) * 1e3)
    if len(hits["x_m"]):
        selected = primary_hits & (hits["event_id"] >= 0) & (hits["event_id"] < len(phase))
        ids = hits["event_id"][selected]
        if ids.size:
            length = (config.geometry.l1_mm + config.geometry.l2_mm) * 1e-3
            bx = phase[ids, 0] + length * momentum[ids, 0] / momentum[ids, 2]
            by = phase[ids, 1] + length * momentum[ids, 1] / momentum[ids, 2]
            shift = np.hypot(hits["x_m"][selected] - bx, hits["y_m"][selected] - by)
            data["median_detector_displacement_um"] = float(np.median(shift) * 1e6)
            undroplet = ~entered_hits[selected]
            data["median_unentered_displacement_um"] = (
                float(np.median(shift[undroplet]) * 1e6) if undroplet.any() else None)
            data["median_entered_displacement_um"] = (
                float(np.median(shift[~undroplet]) * 1e6) if (~undroplet).any() else None)
    return data


def simulate(config: SimulationConfig, output_dir: str | Path | None = None) -> SimulationResult:
    output = Path(output_dir) if output_dir is not None else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    source_rng = np.random.default_rng(config.run.seed)
    phase, momentum = sample_source(config.source, config.run.n_simulated, source_rng)
    if config.run.engine == "geant4":
        if output is None:
            raise ValueError("Geant4 runs require output_dir for source and hit files")
        hits = run_geant4(config, phase, momentum, output / "native")
    else:
        hits = trace_rays(config, build_field(config), phase, momentum)
    images = make_images(config, hits, np.random.default_rng(config.run.seed + 1))
    metrics = _base_metrics(config, hits, images, phase, momentum)
    result = SimulationResult(config, hits, images, metrics, output)
    if output:
        save_result(result)
    return result


def _plot_image(result: SimulationResult, path: Path) -> None:
    image = result.images["observed"]
    edges = result.images["x_edges_m"] * 1e3
    fig, ax = plt.subplots(figsize=(7, 6))
    view = ax.imshow(image, origin="lower", cmap="magma",
                     extent=[edges[0], edges[-1], edges[0], edges[-1]])
    circle = plt.Circle((0, 0), result.config.geometry.radius_m *
                        result.config.geometry.magnification * 1e3,
                        fill=False, color="cyan", linewidth=0.8, alpha=0.75)
    ax.add_patch(circle)
    ax.set(xlabel="Detector x (mm)", ylabel="Detector y (mm)",
           title=f"Electron shadow: Q={result.config.charge.q_e:.2g} e, "
                 f"K={result.config.source.energy_mev:g} MeV")
    fig.colorbar(view, ax=ax, label="Detector signal (gain units/pixel)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def version_fingerprint(config: SimulationConfig) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted([*root.glob("src/droplet_shadow/*.py"), *root.glob("cpp/*.cpp"),
                        *root.glob("cpp/*.txt")]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    fingerprint = {"source_sha256": digest.hexdigest()}
    if config.run.engine == "geant4":
        executable = geant4_executable(config)
        fingerprint["geant4_executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    return fingerprint


def save_result(result: SimulationResult) -> None:
    output = result.output_dir
    if output is None:
        raise ValueError("SimulationResult has no output directory")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "images.npz", **result.images)
    np.savez_compressed(output / "hits.npz", **result.hits)
    versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                **version_fingerprint(result.config)}
    if result.config.run.engine == "geant4":
        config_bin = Path(sys.prefix) / "bin" / "geant4-config"
        if config_bin.is_file():
            versions["geant4"] = subprocess.check_output(
                [str(config_bin), "--version"], text=True).strip()
    details = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "platform": platform.platform(), "versions": versions,
               "config": result.config.to_dict(),
               "metrics": result.metrics,
               "model_caveat": "Local spectroscopic fields are not equated to external Coulomb fields.",
               "mc_sampling_caveat": (
                   "n_simulated < exposure_electrons: the expected intensity may retain transport-MC texture; "
                   "increase n_simulated for quantitative images."
                   if result.config.run.n_simulated < result.config.run.exposure_electrons else None)}
    (output / "result.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_image(result, output / "shadow.png")


def load_images(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.is_dir():
        path = path / "images.npz"
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def compare_result(result: SimulationResult, reference: SimulationResult) -> dict[str, float | None]:
    if not np.array_equal(result.images["x_edges_m"], reference.images["x_edges_m"]):
        raise ValueError("Result and reference pixel grids differ")
    return image_metrics(result.config, result.images["expected"],
                         reference.images["expected"], result.images["x_edges_m"])


def _plot_controls(charged: SimulationResult, neutral: SimulationResult,
                   absent: SimulationResult, path: Path) -> None:
    images = [item.images["expected"] for item in (charged, neutral, absent)]
    edges = charged.images["x_edges_m"]
    extent = np.array([edges[0], edges[-1], edges[0], edges[-1]]) * 1e3
    vmax = np.quantile(np.concatenate([array.ravel() for array in images]), 0.995)
    difference = images[0] - images[1]
    display_difference = gaussian_filter(difference, 3.0)
    vmax_difference = max(float(np.quantile(abs(display_difference), 0.995)), 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, array, name in zip(axes.flat[:2], images[:2], ("Charged water", "Neutral water")):
        shown = ax.imshow(array, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=vmax)
        ax.set(title=name, xlabel="Detector x (mm)", ylabel="Detector y (mm)")
        fig.colorbar(shown, ax=ax, label="Expected signal/pixel")
    shown = axes[1, 0].imshow(display_difference, origin="lower", extent=extent,
                               cmap="coolwarm", vmin=-vmax_difference,
                               vmax=vmax_difference)
    axes[1, 0].set(title="Charged − neutral\n(display blur σ=75 μm)", xlabel="Detector x (mm)",
                   ylabel="Detector y (mm)")
    fig.colorbar(shown, ax=axes[1, 0], label="Expected signal difference/pixel")
    for array, label in zip(images, ("Charged water", "Neutral water", "No droplet")):
        radius, profile = radial_profile(array, edges)
        axes[1, 1].plot(radius * 1e3, profile, label=label)
    axes[1, 1].axvline(charged.config.geometry.radius_m *
                       charged.config.geometry.magnification * 1e3,
                       color="0.5", linestyle="--", linewidth=1)
    axes[1, 1].set(xlabel="Detector radius (mm)",
                   ylabel="Expected signal/pixel", title="Absolute radial intensity")
    axes[1, 1].legend()
    fig.suptitle(f"K={charged.config.source.energy_mev:g} MeV, "
                 f"Q={charged.config.charge.q_e:.2g} e; "
                 f"{charged.config.run.n_simulated:,} simulated electrons")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def simulate_controls(config: SimulationConfig, output_dir: str | Path) -> dict[str, SimulationResult]:
    """Save charged, neutral-material and no-material runs with common source seed."""
    root = Path(output_dir)
    charged = simulate(config, root / "charged")
    neutral_config = config.with_updates(charge={"q_e": 0, "model": "surface",
                                                 "dipole_potential_v": 0,
                                                 "dipole_fraction": 0,
                                                 "quadrupole_fraction": 0,
                                                 "patch_fraction": 0})
    neutral = simulate(neutral_config, root / "neutral")
    absent = simulate(neutral_config.with_updates(run={"droplet_material": "vacuum"}),
                      root / "no_droplet")
    R_projected = config.geometry.radius_m * config.geometry.magnification
    edges = charged.images["x_edges_m"]
    comparison = {"charged_vs_neutral": compare_result(charged, neutral),
                  "neutral_vs_no_droplet": compare_result(neutral, absent),
                  "charged_angular_harmonics": angular_harmonics(
                      charged.images["expected"], edges, R_projected, 1.5 * R_projected),
                  "neutral_angular_harmonics": angular_harmonics(
                      neutral.images["expected"], edges, R_projected, 1.5 * R_projected),
                  "caustic_candidate_impact_um": caustic_locations(config, build_field(config))}
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _plot_controls(charged, neutral, absent, root / "comparison.png")
    return {"charged": charged, "neutral": neutral, "no_droplet": absent}
