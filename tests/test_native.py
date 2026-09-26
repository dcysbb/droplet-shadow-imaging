"""小规模端到端 Geant4 回归测试。

无本机构建时自动跳过；这些测试不只检查程序是否运行，还检查
电荷偏转符号、解析弱偏转量级、静电能量守恒，以及缩小积分步长后
落点变化是否小于 0.01 个探测器像素。
"""

from pathlib import Path

import numpy as np
import pytest

from droplet_shadow.config import SimulationConfig
from droplet_shadow.constants import C, COULOMB_K, E_CHARGE, M_E
from droplet_shadow.native import run_geant4
from droplet_shadow.rays import weak_deflection_angle


EXECUTABLE = Path(__file__).resolve().parents[1] / "build" / "droplet_g4"
pytestmark = pytest.mark.skipif(not EXECUTABLE.exists(), reason="Geant4 core has not been built")


def _single_ray(config: SimulationConfig, path: Path,
                impact_m: float = 100e-6) -> dict[str, np.ndarray]:
    """复制同一条离轴探针给多个事件，检查不同 worker 的场积分一致性。"""
    kinetic = config.source.energy_mev * 1e6 * E_CHARGE
    p = np.sqrt(kinetic * (kinetic + 2 * M_E * C**2)) / C
    phase = np.tile([impact_m, 0, 0, 0], (config.run.n_simulated, 1))
    momentum = np.tile([0, 0, p], (config.run.n_simulated, 1))
    return run_geant4(config, phase, momentum, path)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_geant4_no_material_sign_energy_and_step_convergence(tmp_path, threads):
    """真空对照避免散射随机性，分别检查场物理与数值积分。"""
    base = SimulationConfig().with_updates(
        source={"energy_mev": 1.0},
        run={"engine": "geant4", "droplet_material": "vacuum",
             "n_simulated": threads, "exposure_electrons": threads, "geant4_threads": threads})
    neutral = _single_ray(base.with_updates(charge={"q_e": 0}), tmp_path / "neutral")
    positive_config = base.with_updates(charge={"q_e": 1e7})
    positive = _single_ray(positive_config, tmp_path / "positive")
    positive_volume = _single_ray(positive_config.with_updates(charge={"model": "volume"}),
                                  tmp_path / "positive_volume")
    negative = _single_ray(base.with_updates(charge={"q_e": -1e7}), tmp_path / "negative")
    np.testing.assert_allclose(neutral["x_m"], np.full(threads, 100e-6), atol=1e-10)
    assert np.all(positive["x_m"] < neutral["x_m"])
    assert np.all(neutral["x_m"] < negative["x_m"])
    np.testing.assert_allclose(positive["x_m"], positive_volume["x_m"], atol=1e-12)
    weak_x = 100e-6 + positive_config.geometry.l2_mm * 1e-3 * weak_deflection_angle(
        1e7, 100e-6, 1.0)
    assert np.all(abs(positive["x_m"] - weak_x) < 2e-6)
    phi_start = COULOMB_K * 1e7 * E_CHARGE / np.hypot(100e-6, 10e-3)
    phi_end = COULOMB_K * 1e7 * E_CHARGE / np.hypot(positive["x_m"], 0.5)
    expected_energy = 1.0 + (phi_end - phi_start) * 1e-6
    assert np.all(abs(positive["energy_mev"] - expected_energy) < 2e-8)
    refined = _single_ray(positive_config.with_updates(run={
        "geant4_world_step_mm": 2.5, "geant4_far_step_mm": 0.125,
        "geant4_near_step_um": 5.0}), tmp_path / "refined")
    pixel_m = positive_config.detector.pixel_um * 1e-6
    assert np.all(abs(refined["x_m"] - positive["x_m"]) < 0.01 * pixel_m)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_geant4_thin_dipole_layer_guard_and_step_convergence(tmp_path, threads):
    """电子掠过纳米层时，边界保护区应防止虚假的能量增减。"""
    base = SimulationConfig().with_updates(
        charge={"q_e": 0, "dipole_potential_v": 1.0},
        run={"engine": "geant4", "droplet_material": "vacuum",
             "n_simulated": threads, "exposure_electrons": threads, "geant4_threads": threads})
    for offset_nm in (0.25, 0.75):
        impact = base.geometry.radius_m - offset_nm * 1e-9
        ordinary = _single_ray(base, tmp_path / f"ordinary_{offset_nm}", impact)
        refined = _single_ray(base.with_updates(run={"geant4_layer_step_scale": 0.5}),
                              tmp_path / f"refined_{offset_nm}", impact)
        for output in (ordinary, refined):
            assert np.all(abs(output["energy_mev"] - base.source.energy_mev) < 1e-8)
        assert np.all(abs(ordinary["x_m"] - refined["x_m"]) < 0.01 * base.detector.pixel_um * 1e-6)


@pytest.mark.parametrize("model", ["volume", "poisson_boltzmann"])
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_geant4_interior_radial_field_preserves_electrostatic_energy(tmp_path, model, threads):
    """穿滴真空电子直接检查球内场与球外尾场的接续，包括 PB 插值。"""
    from droplet_shadow.fields import build_field
    charge = {"model":model,"q_e":1e6}
    if model == "poisson_boltzmann":
        charge.update(surface_fixed_e=-1e6,ion_positive_count=2e6)
    cfg = SimulationConfig().with_updates(charge=charge,
        run={"engine":"geant4","droplet_material":"vacuum","n_simulated":threads,
             "exposure_electrons":threads,"geant4_threads":threads})
    b = 20e-6
    hits = _single_ray(cfg, tmp_path/model, b)
    assert hits['x_m'].size == threads
    field = build_field(cfg)
    phi0 = field.potential(np.array([[b,0,-cfg.geometry.l1_mm*1e-3]]))[0]
    phi1 = field.potential(np.column_stack((hits['x_m'], hits['y_m'],
                                           np.full(threads, cfg.geometry.l2_mm*1e-3))))
    expected = cfg.source.energy_mev+(phi1-phi0)*1e-6
    assert np.all(abs(hits['energy_mev']-expected) < 2e-8)
