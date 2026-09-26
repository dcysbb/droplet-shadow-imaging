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


@pytest.mark.parametrize("positive,negative,fixed", [(1e6, 0, -1e6), (1.1e6, 1e5, -1e6), (1e6, 0, -9e5)])
def test_pb_exact_surface_gauss_and_potential_gradient(positive, negative, fixed):
    """独立用高斯定律检查 PB 内场，并检查查表电场确为电势负梯度。"""
    from droplet_shadow.constants import COULOMB_K
    cfg = SimulationConfig().with_updates(charge={
        "model": "poisson_boltzmann", "q_e": fixed + positive - negative,
        "surface_fixed_e": fixed, "ion_positive_count": positive, "ion_negative_count": negative})
    field = poisson_boltzmann(cfg)
    R = cfg.geometry.radius_m
    assert field.radii_m[0] == 0 and field.radii_m[-1] == R
    np.testing.assert_allclose(field.ion_counts, [n for n in (positive, negative) if n > 0], rtol=1e-9)
    np.testing.assert_allclose(field.field_v_m[-1], COULOMB_K * E_CHARGE * (positive-negative) /
                               (cfg.geometry.epsilon_water*R**2), rtol=2e-6)
    at_surface = field.electric_field(np.array([[R, 0, 0]]))[0, 0]
    np.testing.assert_allclose(at_surface, COULOMB_K*cfg.charge.q_e*E_CHARGE/R**2, atol=1e-9)
    r = R*np.array([0.1, 0.5, 0.9, 0.99])
    p = np.column_stack((r, np.zeros_like(r), np.zeros_like(r)))
    h = R*1e-6
    gradient = (field.potential(p+[h,0,0])-field.potential(p-[h,0,0]))/(2*h)
    np.testing.assert_allclose(field.electric_field(p)[:,0], -gradient, rtol=2e-5, atol=1e-5)
    refined = poisson_boltzmann(cfg, nodes=640, tol=1e-7)
    np.testing.assert_allclose(field.electric_field(p), refined.electric_field(p), rtol=2e-4, atol=1e-4)


def test_shared_dielectric_layer_obeys_gauss_and_prescribed_dipole_voltage():
    """体电荷在偶极介电层内也必须用层介电常数，不能继续使用水的 78。"""
    from droplet_shadow.constants import COULOMB_K
    cfg = SimulationConfig().with_updates(charge={
        "model": "volume", "q_e": 1e6, "dipole_potential_v": 1,
        "dipole_thickness_nm": 5000, "dipole_epsilon": 2})
    R = cfg.geometry.radius_m
    a = R-5000e-9
    r = (R+a)/2
    layer_charge = 2*a*R/(COULOMB_K*(R-a))
    enclosed = cfg.charge.q_e*E_CHARGE*(r/R)**3 + layer_charge
    f = build_field(cfg)
    np.testing.assert_allclose(f.electric_field(np.array([[r,0,0]]))[0,0],
                               enclosed/(4*np.pi*EPS0*2*r**2), rtol=1e-12)
    # 单独等效偶极层仍保持给定 1 V 势差，且没有外部电场。
    dipole = build_field(cfg.with_updates(charge={"q_e":0}))
    np.testing.assert_allclose(dipole.potential(np.array([[0,0,0],[2*R,0,0]])), [1,0], atol=1e-12)
    # 独立有限差分检查组合场的电势导数。
    h = 1e-10
    gradient = (f.potential(np.array([[r+h,0,0]]))-f.potential(np.array([[r-h,0,0]])))/(2*h)
    np.testing.assert_allclose(-gradient[0], f.electric_field(np.array([[r,0,0]]))[0,0], rtol=1e-8)


def test_direct_sum_rejects_pb_and_inconsistent_media():
    from droplet_shadow.fields import SumField, RadialShells
    cfg = SimulationConfig().with_updates(charge={"model":"poisson_boltzmann", "q_e":0,
        "surface_fixed_e":-1e6, "ion_positive_count":1e6})
    with pytest.raises(ValueError, match="coupled ion re-solve"):
        SumField(build_field(cfg), build_field(SimulationConfig()))
    R = cfg.geometry.radius_m
    with pytest.raises(ValueError, match="dielectric profile"):
        SumField(RadialShells(R,(R,),(1e-13,),78), RadialShells(R,(R,),(1e-13,),2))


def test_pb_surface_charge_offset_does_not_change_mobile_ion_solution():
    """固定离子数不变时，增加均匀表面电荷只给球内势增加常数。"""
    from droplet_shadow.constants import COULOMB_K
    cfg = SimulationConfig().with_updates(charge={"model":"poisson_boltzmann", "q_e":0,
        "surface_fixed_e":-1e6, "ion_positive_count":1e6})
    baseline = poisson_boltzmann(cfg)
    charged = poisson_boltzmann(cfg.with_updates(charge={"q_e":1e7,"surface_fixed_e":9e6}))
    R = cfg.geometry.radius_m
    p = np.array([[0.2*R,0,0],[0.5*R,0,0],[0.9*R,0,0]])
    np.testing.assert_allclose(charged.electric_field(p), baseline.electric_field(p), rtol=1e-7, atol=1e-5)
    np.testing.assert_allclose(charged.potential(p)-baseline.potential(p),
                               COULOMB_K*1e7*E_CHARGE/R, rtol=1e-12)


def test_double_layer_not_added_twice():
    cfg = SimulationConfig().with_updates(charge={"q_e":1e6,"double_layer_q_e":1e8})
    p = np.array([[0,0,0],[cfg.geometry.radius_m-0.5e-9,0,0],[1e-4,0,0]])
    first = build_field(cfg)
    second = build_field(cfg.with_updates(charge={"model":"neutral_double_layer"}))
    np.testing.assert_allclose(first.potential(p), second.potential(p), rtol=1e-12)
    np.testing.assert_allclose(first.electric_field(p), second.electric_field(p), rtol=1e-12)


def test_removed_charge_parameters_are_not_silently_ignored():
    with pytest.raises(TypeError, match="dipole_fraction"):
        SimulationConfig().with_updates(charge={"dipole_fraction":1})
