"""球对称模型到输运接口的回归：边界跳变和无场对照必须正确。"""

import numpy as np
import pytest

from droplet_shadow import SimulationConfig, build_field
from droplet_shadow.native import export_field
from droplet_shadow.simulation import simulate_controls


@pytest.mark.parametrize("model", ["surface", "volume", "neutral_double_layer", "poisson_boltzmann"])
def test_export_keeps_two_sides_of_surface(tmp_path, model):
    charge = {"model":model, "q_e":1e6}
    if model == "poisson_boltzmann":
        charge.update(surface_fixed_e=-1e6, ion_positive_count=2e6)
    if model == "neutral_double_layer":
        charge.update(double_layer_q_e=1e8)
    cfg = SimulationConfig().with_updates(charge=charge)
    path = tmp_path/'field.csv'
    export_field(cfg, path)
    table = np.genfromtxt(path, delimiter=',', skip_header=1)
    R = cfg.geometry.radius_m
    assert table[-1,0] == table[-2,0] == R*1e3
    np.testing.assert_allclose(table[-1,1], table[-2,1], atol=0)
    f = build_field(cfg)
    points = np.array([[np.nextafter(R,0),0,0],[R,0,0]])
    np.testing.assert_allclose(table[-2:,2]*1e3, f.electric_field(points)[:,0], rtol=1e-12)
    assert np.all(np.diff(table[:,0]) >= 0)


@pytest.mark.parametrize("charge", [
    {"model":"neutral_double_layer","q_e":0,"double_layer_q_e":1e8},
    {"model":"poisson_boltzmann","q_e":0,"surface_fixed_e":-1e6,"ion_positive_count":1e6},
])
def test_controls_clear_all_field_sources(tmp_path, charge):
    cfg = SimulationConfig().with_updates(charge=charge,
        run={"engine":"ray","n_simulated":8,"exposure_electrons":8,"raster_width_mm":2},
        detector={"pixel_um":100})
    controls = simulate_controls(cfg, tmp_path)
    neutral = controls['neutral'].config
    points = np.array([[0,0,0],[cfg.geometry.radius_m-0.5e-9,0,0],[1e-4,0,0]])
    np.testing.assert_allclose(build_field(neutral).electric_field(points),0,atol=0)
    assert neutral.charge.ion_positive_count == neutral.charge.ion_negative_count == 0
    assert neutral.charge.double_layer_q_e == neutral.charge.dipole_potential_v == 0
