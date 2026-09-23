"""纯 Python 物理不变量：高斯定律、球外简并、PB 守恒和源发射度。

这些单元测试大多不需要 Geant4；读者可从测试名和断言反推
各模型“必须成立”的最小物理事实。测试容差表示数值精度要求，
不是实验测量不确定度。
"""

import numpy as np
import pytest

from droplet_shadow.analysis import calibrated_detection_limit
from droplet_shadow.config import SimulationConfig
from droplet_shadow.constants import E_CHARGE, EPS0
from droplet_shadow.fields import build_field, poisson_boltzmann, rayleigh_charge_e
from droplet_shadow.rays import trace_rays, weak_deflection_angle
from droplet_shadow.source import geometric_emittance, phase_space_covariance


def test_surface_and_volume_share_exterior_but_not_interior():
    base = SimulationConfig().with_updates(charge={"q_e": 1e6})
    shell = build_field(base)
    bulk = build_field(base.with_updates(charge={"model": "volume"}))
    exterior = np.array([[80e-6, 10e-6, 0]])
    interior = np.array([[10e-6, 0, 0]])
    np.testing.assert_allclose(shell.electric_field(exterior), bulk.electric_field(exterior), rtol=1e-12)
    assert not np.allclose(shell.electric_field(interior), bulk.electric_field(interior))


def test_neutral_double_layer_and_dipole_layer_have_zero_exterior_field():
    cfg = SimulationConfig().with_updates(charge={
        "model": "neutral_double_layer", "q_e": 0,
        "double_layer_q_e": 1e8, "double_layer_thickness_nm": 2,
        "dipole_potential_v": 1.0, "dipole_thickness_nm": 0.5,
    })
    field = build_field(cfg)
    outside = np.array([[80e-6, 0, 0], [0, 0, 100e-6]])
    np.testing.assert_allclose(field.electric_field(outside), 0, atol=1e-10)
    np.testing.assert_allclose(field.potential(outside), 0, atol=1e-10)
    pure_dipole = build_field(cfg.with_updates(charge={"double_layer_q_e": 0}))
    inner = np.array([[0, 0, 0]])
    assert abs(pure_dipole.potential(inner)[0] - 1.0) < 1e-9


def test_poisson_boltzmann_conserves_ions_and_total_charge():
    cfg = SimulationConfig().with_updates(charge={
        "model": "poisson_boltzmann", "q_e": 0,
        "surface_fixed_e": -1e6, "ion_positive_count": 1e6,
        "ion_negative_count": 0,
    })
    field = poisson_boltzmann(cfg)
    np.testing.assert_allclose(field.electric_field(np.array([[100e-6, 0, 0]])), 0, atol=1e-9)
    assert abs(field.field_v_m[0]) < 1e-4
    assert field.field_v_m[-1] > 0


def test_source_covariance_sets_emittance():
    cfg = SimulationConfig().with_updates(source={"x_xprime_correlation": 0.6})
    cov = phase_space_covariance(cfg.source)
    ex, ey = geometric_emittance(cfg.source)
    np.testing.assert_allclose(ex, np.sqrt(cov[0, 0] * cov[2, 2]) * 0.8)
    np.testing.assert_allclose(ey, np.sqrt(cov[1, 1] * cov[3, 3]))


def test_weak_deflection_sign_and_ray_order_of_magnitude():
    cfg = SimulationConfig().with_updates(
        source={"energy_mev": 1.0}, charge={"q_e": 1e6},
        run={"engine": "ray", "n_simulated": 1, "exposure_electrons": 1})
    b = 100e-6
    phase = np.array([[b, 0, 0, 0]])
    kinetic = 1e6 * E_CHARGE
    from droplet_shadow.constants import M_E, C
    p = np.sqrt(kinetic * (kinetic + 2 * M_E * C**2)) / C
    result = trace_rays(cfg, build_field(cfg), phase, np.array([[0, 0, p]]))
    angle = weak_deflection_angle(cfg.charge.q_e, b, 1.0)
    assert result["x_m"][0] < b
    assert abs((result["x_m"][0] - b) / (cfg.geometry.l2_mm * 1e-3 * angle) - 1) < 0.15


def test_rayleigh_fraction_for_reference_case():
    cfg = SimulationConfig()
    limit = rayleigh_charge_e(cfg.geometry.radius_m, cfg.geometry.surface_tension_n_m)
    assert 4e7 < limit < 5e7


def test_gauss_and_dielectric_boundary_conditions():
    cfg = SimulationConfig().with_updates(charge={"q_e": 1e6})
    R = cfg.geometry.radius_m
    delta = 1e-10
    p = np.array([[R - delta, 0, 0], [R + delta, 0, 0]])
    surface = build_field(cfg).electric_field(p)[:, 0]
    volume = build_field(cfg.with_updates(charge={"model": "volume"})).electric_field(p)[:, 0]
    sigma = cfg.charge.q_e * E_CHARGE / (4 * np.pi * R**2)
    np.testing.assert_allclose(EPS0 * (surface[1] - cfg.geometry.epsilon_water * surface[0]),
                               sigma, rtol=1e-5)
    np.testing.assert_allclose(EPS0 * (volume[1] - cfg.geometry.epsilon_water * volume[0]),
                               0, atol=2e-11)
    assert abs(surface[1] * 4 * np.pi * EPS0 * (R + delta)**2 / E_CHARGE - 1e6) < 1e-5


def test_asymmetric_surface_charge_has_zero_extra_monopole():
    cfg = SimulationConfig().with_updates(charge={
        "q_e": 1e6, "dipole_fraction": 0.5, "quadrupole_fraction": 0.2,
        "patch_fraction": 0.3, "multipole_order": 8})
    field = build_field(cfg)
    R = cfg.geometry.radius_m
    points = np.array([[2*R, 0, 0], [-2*R, 0, 0], [0, 2*R, 0]])
    potential = field.potential(points)
    assert abs(potential[0] - potential[1]) > 0.01
    # The extra non-monopole field vanishes faster than 1/r².
    reference = build_field(cfg.with_updates(charge={
        "dipole_fraction": 0, "quadrupole_fraction": 0, "patch_fraction": 0}))
    near = (field.electric_field(np.array([[2*R, 0, 0]])) -
            reference.electric_field(np.array([[2*R, 0, 0]])))[0, 0]
    far = (field.electric_field(np.array([[10*R, 0, 0]])) -
           reference.electric_field(np.array([[10*R, 0, 0]])))[0, 0]
    assert abs(near) > 50 * abs(far)


def test_empirical_five_percent_ninety_five_percent_detection_rule():
    null = np.arange(100, dtype=float)
    alternatives = {1e4: np.full(100, 90.0),
                    1e5: np.r_[np.full(4, 0.0), np.full(96, 200.0)]}
    assert calibrated_detection_limit(null, alternatives) == 1e5


def test_pb_refuses_uncoupled_polarization_superposition():
    with pytest.raises(ValueError, match="coupled ion re-solve"):
        SimulationConfig().with_updates(charge={
            "model": "poisson_boltzmann", "q_e": 0,
            "surface_fixed_e": -1e6, "ion_positive_count": 1e6,
            "dipole_potential_v": 1.0})
