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
    """用一条离轴、零发散的已知动量电子隔离电场积分误差。"""
    kinetic = config.source.energy_mev * 1e6 * E_CHARGE
    p = np.sqrt(kinetic * (kinetic + 2 * M_E * C**2)) / C
    phase = np.array([[impact_m, 0, 0, 0]])
    momentum = np.array([[0, 0, p]])
    return run_geant4(config, phase, momentum, path)


def test_geant4_no_material_sign_energy_and_step_convergence(tmp_path):
    """真空对照避免散射随机性，分别检查场物理与数值积分。"""
    base = SimulationConfig().with_updates(
        source={"energy_mev": 1.0},
        run={"engine": "geant4", "droplet_material": "vacuum",
             "n_simulated": 1, "exposure_electrons": 1})
    neutral = _single_ray(base.with_updates(charge={"q_e": 0}), tmp_path / "neutral")
    positive_config = base.with_updates(charge={"q_e": 1e7})
    positive = _single_ray(positive_config, tmp_path / "positive")
    positive_volume = _single_ray(positive_config.with_updates(charge={"model": "volume"}),
                                  tmp_path / "positive_volume")
    negative = _single_ray(base.with_updates(charge={"q_e": -1e7}), tmp_path / "negative")
    np.testing.assert_allclose(neutral["x_m"], [100e-6], atol=1e-10)
    assert positive["x_m"][0] < neutral["x_m"][0] < negative["x_m"][0]
    np.testing.assert_allclose(positive["x_m"], positive_volume["x_m"], atol=1e-12)
    weak_x = 100e-6 + positive_config.geometry.l2_mm * 1e-3 * weak_deflection_angle(
        1e7, 100e-6, 1.0)
    assert abs(positive["x_m"][0] - weak_x) < 2e-6
    phi_start = COULOMB_K * 1e7 * E_CHARGE / np.hypot(100e-6, 10e-3)
    phi_end = COULOMB_K * 1e7 * E_CHARGE / np.hypot(positive["x_m"][0], 0.5)
    expected_energy = 1.0 + (phi_end - phi_start) * 1e-6
    assert abs(positive["energy_mev"][0] - expected_energy) < 2e-8
    refined = _single_ray(positive_config.with_updates(run={
        "geant4_world_step_mm": 2.5, "geant4_far_step_mm": 0.125,
        "geant4_near_step_um": 5.0}), tmp_path / "refined")
    pixel_m = positive_config.detector.pixel_um * 1e-6
    assert abs(refined["x_m"][0] - positive["x_m"][0]) < 0.01 * pixel_m


def test_geant4_thin_dipole_layer_guard_and_step_convergence(tmp_path):
    """电子掠过纳米层时，边界保护区应防止虚假的能量增减。"""
    base = SimulationConfig().with_updates(
        charge={"q_e": 0, "dipole_potential_v": 1.0},
        run={"engine": "geant4", "droplet_material": "vacuum",
             "n_simulated": 1, "exposure_electrons": 1})
    for offset_nm in (0.25, 0.75):
        impact = base.geometry.radius_m - offset_nm * 1e-9
        ordinary = _single_ray(base, tmp_path / f"ordinary_{offset_nm}", impact)
        refined = _single_ray(base.with_updates(run={"geant4_layer_step_scale": 0.5}),
                              tmp_path / f"refined_{offset_nm}", impact)
        for output in (ordinary, refined):
            assert abs(output["energy_mev"][0] - base.source.energy_mev) < 1e-8
        assert abs(ordinary["x_m"][0] - refined["x_m"][0]) < 0.01 * base.detector.pixel_um * 1e-6
