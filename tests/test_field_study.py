"""本次交付唯一指定的轻量测试入口；禁止真实 Geant4 和 ray 输运。

假落点只用于隔离验证数据处理，全部写 pytest 临时目录；它们绝不作为
研究结果或“已运行”状态交付。默认入口测试也断言没有隐式启动计算。
"""

from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from droplet_shadow import SimulationConfig, build_field
from droplet_shadow.analysis import spatial_resolution_object_um
from droplet_shadow.config import Detector
from droplet_shadow.detector import make_images
from droplet_shadow.fields import poisson_boltzmann
from droplet_shadow.study import (ROOT, REQUIRED_OUTPUTS, completed_path, config_from_dict,
                                   ion_count, load_study, planned_cases, run_case, study_lock, validate_output)
from droplet_shadow.study_analysis import (analyze_case, analyze_study, classified_profiles,
                                           compatible_reference, field_figure, comparison_figures, focus_figure,
                                           focus_radial_figure)


@pytest.fixture(autouse=True)
def forbid_transport(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real electron transport is forbidden in the delivery tests")
    import droplet_shadow.simulation as simulation
    import droplet_shadow.native as native
    import droplet_shadow.rays as rays
    monkeypatch.setattr(simulation, "simulate", forbidden)
    monkeypatch.setattr(simulation, "run_geant4", forbidden)
    monkeypatch.setattr(simulation, "trace_rays", forbidden)
    monkeypatch.setattr(native, "run_geant4", forbidden)
    monkeypatch.setattr(native, "_run_native_process", forbidden)
    monkeypatch.setattr(rays, "trace_rays", forbidden)


@pytest.fixture
def small_case():
    case = planned_cases()[1]
    cfg = case.config.with_updates(
        detector={"pixel_um": 100, "diameter_mm": 2},
        run={"n_simulated": 20, "exposure_electrons": 20, "raster_width_mm": 2, "show_progress": False})
    return replace(case, config=cfg)


def toy_hits():
    # event 0 含一个初级和一个次级；另外 17 个初级没有落点，仍进入分母。
    return dict(x_m=np.array([.1, .2, .3, 2.0]) * 1e-3, y_m=np.zeros(4),
                event_id=np.array([0, 0, 3, 5]), track_id=np.array([1, 2, 1, 1]),
                parent_id=np.array([0, 1, 0, 0]), energy_mev=np.full(4, 3.),
                primary=np.array([True, False, True, True]), entered=np.array([True, True, False, False]),
                blocked=np.zeros(4, dtype=bool), deposited_e_by_event=np.zeros(20))


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_psf(value):
    with pytest.raises(ValueError, match="psf"):
        Detector(psf_fwhm_um=value)


def test_zero_psf_no_convolution_aliasing_or_spreading(small_case, monkeypatch):
    import droplet_shadow.detector as detector
    monkeypatch.setattr(detector, "gaussian_filter", lambda *a, **k: pytest.fail("PSF=0 called convolution"))
    hits = {"x_m": np.array([0.]), "y_m": np.array([0.]), "blocked": np.array([False])}
    result = make_images(small_case.config, hits, np.random.default_rng(1))
    assert np.count_nonzero(result["expected"]) == 1
    assert result["expected"].sum() == 1
    assert np.count_nonzero(result["observed"]) <= 1
    for a, b in (("ideal", "expected"), ("expected", "observed"), ("ideal", "observed")):
        assert not np.shares_memory(result[a], result[b])
    result["observed"][:] = -999
    assert result["expected"].sum() == 1
    assert Detector().psf_fwhm_um == 50


def test_positive_psf_original_gaussian(small_case):
    from scipy.ndimage import gaussian_filter
    from droplet_shadow.constants import FWHM_SIGMA
    cfg = small_case.config.with_updates(detector={"psf_fwhm_um": 200})
    images = make_images(cfg, toy_hits(), np.random.default_rng(1))
    np.testing.assert_allclose(images["expected"], gaussian_filter(images["ideal"], 2 / FWHM_SIGMA, mode="constant"))
    assert np.count_nonzero(images["expected"]) > np.count_nonzero(images["ideal"])


def test_other_positive_parameters_still_required():
    for key in ("pixel_um", "diameter_mm", "gain_mean", "gain_shape"):
        with pytest.raises(ValueError):
            Detector(psf_fwhm_um=0, **{key: 0})


@pytest.mark.parametrize("focus,expected", [(10, 9.950608), (5, 4.975843), (1, .998611)])
def test_resolution_with_pixels_without_psf(focus, expected):
    cfg = planned_cases()[0].config.with_updates(source={"crossover_fwhm_um": focus})
    assert spatial_resolution_object_um(cfg) == pytest.approx(expected, abs=5e-7)


def test_complete_matrix_and_generated_configs():
    generated, saved = planned_cases(), load_study()
    assert generated == saved
    assert len(saved) == len({c.case_id for c in saved}) == 35
    assert [sum(c.group == g for c in saved) for g in ("main", "model_extra", "distance_extra")] == [24, 6, 5]
    by_id = {c.case_id: c for c in saved}
    for case in saved:
        cfg = case.config
        assert cfg.run.n_simulated == 10_000_000
        assert cfg.detector.psf_fwhm_um == 0
        assert cfg.run.geant4_executable is None
        assert cfg.source.divergence_rms_mrad == 10
        assert config_from_dict(cfg.to_dict()) == cfg
        compatible_reference(case, by_id[case.neutral_case])
        compatible_reference(case, by_id[case.absent_case])
    manifest = json.loads((ROOT / "examples/field_scale_study/manifest.json").read_text())
    assert all(c["status"] == "not_run" for c in manifest["cases"])


def test_concentration_units_and_pb_conservation():
    assert ion_count(1, 50e-6) == pytest.approx(315318552.8416609)
    for case in planned_cases():
        if case.model.startswith("pb_") and case.config.source.crossover_fwhm_um == 10 and case.group == "main":
            cfg = case.config
            assert cfg.charge.surface_fixed_e + cfg.charge.ion_positive_count == 0
            field = poisson_boltzmann(cfg)
            np.testing.assert_allclose(field.ion_counts, [cfg.charge.ion_positive_count], rtol=1e-8)
            np.testing.assert_allclose(field.electric_field(np.array([[100e-6, 0, 0]])), 0, atol=1e-10)


def test_neutral_layers_external_field_and_potential():
    for case in planned_cases():
        if case.model in ("dipole_0p5nm", "dipole_1nm", "shell_10nm"):
            f = build_field(case.config)
            np.testing.assert_allclose(f.electric_field(np.array([[100e-6, 0, 0]])), 0, atol=1e-8)
            if case.model.startswith("dipole"):
                assert f.potential(np.array([[0., 0., 0.]]))[0] == pytest.approx(1, abs=1e-6)


def test_classes_subsets_and_missing_primary_denominators(small_case):
    p = classified_profiles(small_case.config, toy_hits())
    assert p["all_counts"].sum() == 3  # 第四条在探测器以外。
    np.testing.assert_array_equal(p["all_counts"], sum(p[k + "_counts"] for k in (
        "primary_unentered", "primary_entered", "secondary")))
    np.testing.assert_array_equal(p["subset_counts"].sum(axis=0), p["all_counts"])
    assert p["subset_primary_counts"].sum() == 20
    assert np.sum(p["all_per_primary_mm2"] * p["annulus_area_mm2"]) == pytest.approx(3 / 20)
    weighted = np.average(p["subset_per_primary_mm2"], weights=p["subset_primary_counts"], axis=0)
    np.testing.assert_allclose(weighted, p["all_per_primary_mm2"])


def test_zero_hits_are_valid(small_case):
    hits = toy_hits()
    hits = {k: v if k == "deposited_e_by_event" else v[:0] for k, v in hits.items()}
    profiles = classified_profiles(small_case.config, hits)
    assert profiles["all_counts"].sum() == 0


def test_invalid_event_id_rejected(small_case):
    hits = toy_hits()
    hits["event_id"][0] = 20
    with pytest.raises(ValueError):
        classified_profiles(small_case.config, hits)


def test_three_exposures_only_reweight_raw_hits(small_case, tmp_path):
    attempt = tmp_path / "input"
    attempt.mkdir()
    np.savez_compressed(attempt / "hits.npz", **toy_hits())
    out = tmp_path / "analysis"
    analyze_case(small_case, attempt, out)
    totals = []
    for exposure in (100_000, 1_000_000, 10_000_000):
        with np.load(out / f"exposure_{exposure}.npz") as images:
            totals.append(images["expected"].sum())
    np.testing.assert_allclose(totals, np.array([100_000, 1_000_000, 10_000_000]) * 3 / 20)
    assert (out / "profiles.csv").exists()


def fake_simulator(cfg, destination):
    """替身只写临时测试文件，绝不产生实际粒子轨迹。"""
    for name in REQUIRED_OUTPUTS:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name != "config.yaml":
            path.write_text("TEST FIXTURE — NOT A SIMULATION", encoding="utf-8")
    (destination / "result.json").write_text(json.dumps({"config": cfg.to_dict(), "versions": {}}))
    (destination / "native/transport.json").write_text(json.dumps({"workers": [{"id": 0, "events": cfg.run.n_simulated}]}))
    np.savez_compressed(destination / "hits.npz", **toy_hits())
    np.savez_compressed(destination / "images.npz", **make_images(cfg, toy_hits(), np.random.default_rng(2)))


@pytest.fixture
def mock_execution(monkeypatch):
    monkeypatch.setattr("droplet_shadow.study.execution_fingerprint", lambda c: {"test_fingerprint": "v1"})
    monkeypatch.setattr("droplet_shadow.simulation.simulate", fake_simulator)


def test_resume_validates_and_does_not_rerun(mock_execution, small_case, tmp_path, monkeypatch):
    first = run_case(small_case, tmp_path)
    monkeypatch.setattr("droplet_shadow.simulation.simulate", lambda *a: pytest.fail("cached transport rerun"))
    assert run_case(small_case, tmp_path, resume=True) == first
    assert completed_path(small_case, tmp_path) == first
    with pytest.raises(FileExistsError):
        run_case(small_case, tmp_path)


@pytest.mark.parametrize("kind", ["file", "config", "version"])
def test_stale_or_corrupt_cache_rejected(mock_execution, small_case, tmp_path, monkeypatch, kind):
    path = run_case(small_case, tmp_path)
    if kind == "file":
        (path / "native/hits.csv").write_text("modified")
    elif kind == "config":
        small_case = replace(small_case, config=small_case.config.with_updates(source={"energy_mev": 2}))
    else:
        monkeypatch.setattr("droplet_shadow.study.execution_fingerprint", lambda c: {"test_fingerprint": "v2"})
    with pytest.raises(RuntimeError):
        run_case(small_case, tmp_path, resume=True)


def test_failed_attempt_preserved_and_retried(mock_execution, small_case, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected failure")
    monkeypatch.setattr("droplet_shadow.simulation.simulate", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run_case(small_case, tmp_path)
    assert completed_path(small_case, tmp_path) is None
    state_path = tmp_path / small_case.case_id / "attempt-0001/attempt_state.json"
    assert json.loads(state_path.read_text())["status"] == "failed"
    monkeypatch.setattr("droplet_shadow.simulation.simulate", fake_simulator)
    assert run_case(small_case, tmp_path, resume=True).name == "attempt-0002"
    assert state_path.is_file()


def test_incomplete_output_never_complete(mock_execution, small_case, tmp_path, monkeypatch):
    monkeypatch.setattr("droplet_shadow.simulation.simulate", lambda *a: None)
    with pytest.raises(RuntimeError, match="Missing output"):
        run_case(small_case, tmp_path)
    assert completed_path(small_case, tmp_path) is None


def test_lock_refuses_second_writer(tmp_path):
    with study_lock(tmp_path):
        with pytest.raises(FileExistsError):
            with study_lock(tmp_path):
                pytest.fail("second writer entered")
    assert not (tmp_path / ".study.lock").exists()


def test_missing_results_do_not_trigger_transport(tmp_path):
    status = analyze_study(load_study(), tmp_path)
    assert len(status) == 35
    assert all(s.startswith("missing") for s in status.values())


@pytest.mark.parametrize("filename", ["run_field_scale_study.py", "validate_field_study.py"])
def test_default_cli_never_transports(filename, capsys):
    spec = importlib.util.spec_from_file_location("test_cli", ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main([])
    assert "CHECK ONLY" in capsys.readouterr().out


def test_field_plots_linear_and_separate():
    for case in planned_cases()[:8]:
        fig = field_figure(case)
        assert len(fig.axes) == 4
        assert all(ax.get_xscale() == ax.get_yscale() == "linear" for ax in fig.axes)
        plt.close(fig)


def load_tool(filename):
    spec = importlib.util.spec_from_file_location("delivery_tool", ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("args", [["--run"], ["--resume"], ["--case", "unknown"]])
def test_cli_requires_explicit_valid_selection(args, tmp_path):
    output = tmp_path / "not_created"
    with pytest.raises(SystemExit) as error:
        load_tool("run_field_scale_study.py").main([*args, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("damage", ["event_count", "deposition", "event_id", "image"])
def test_semantic_output_validation(mock_execution, small_case, tmp_path, damage):
    path = run_case(small_case, tmp_path)
    if damage == "event_count":
        (path / "native/transport.json").write_text(json.dumps({"workers": [{"events": 19}]}))
    elif damage in ("deposition", "event_id"):
        hits = toy_hits()
        if damage == "deposition":
            hits["deposited_e_by_event"] = np.zeros(19)
        else:
            hits["event_id"][0] = 20
        np.savez_compressed(path / "hits.npz", **hits)
    else:
        np.savez_compressed(path / "images.npz", **{k: np.zeros((2, 2)) for k in ("ideal", "expected", "observed")})
    with pytest.raises(RuntimeError):
        validate_output(path, small_case.config)


def test_comparison_flux_scales_and_unsmoothed_difference(small_case, tmp_path):
    model = replace(small_case, case_id="model", model="test-model", neutral_case="neutral", absent_case="absent")
    neutral = replace(small_case, case_id="neutral")
    absent = replace(small_case, case_id="absent")
    arrays = []
    for case, factor in ((model, 1), (neutral, 2), (absent, 3)):
        target = tmp_path / case.case_id
        target.mkdir()
        data = make_images(case.config, toy_hits(), np.random.default_rng(1))
        data["expected"] *= factor
        arrays.append(data["expected"])
        np.savez_compressed(target / "exposure_1000000.npz", **data)
    fig, radial = comparison_figures(model, neutral, absent, tmp_path)
    plots = [ax.images[0] for ax in fig.axes if ax.images]
    assert plots[0].get_clim() == plots[1].get_clim() == plots[2].get_clim()
    np.testing.assert_array_equal(plots[3].get_array(), arrays[0] - arrays[1])
    assert len(radial.axes[0].lines) == 3
    plt.close(fig)
    plt.close(radial)


def test_focus_plot_uses_shared_scales(small_case, tmp_path):
    cases = []
    for focus in (10, 5, 1):
        case = replace(small_case, case_id=f"model{focus}", model="same-model", neutral_case=f"neutral{focus}",
                       config=small_case.config.with_updates(source={"crossover_fwhm_um": focus}))
        cases.append(case)
        for name, factor in ((case.case_id, focus), (case.neutral_case, .5)):
            target = tmp_path / name
            target.mkdir()
            data = make_images(case.config, toy_hits(), np.random.default_rng(1))
            data["expected"] *= factor
            np.savez_compressed(target / "exposure_1000000.npz", **data)
    fig = focus_figure(cases, tmp_path)
    plots = [ax.images[0] for ax in fig.axes if ax.images]
    assert plots[0].get_clim() == plots[1].get_clim() == plots[2].get_clim()
    assert plots[3].get_clim() == plots[4].get_clim() == plots[5].get_clim()
    plt.close(fig)
    fig = focus_figure(cases, tmp_path, edge_zoom=True)
    assert fig.axes[0].get_xlim() == pytest.approx((30 * 201 / 1000, 70 * 201 / 1000))
    plt.close(fig)
    fig = focus_radial_figure(cases, tmp_path)
    assert all(len(ax.lines) == 3 for ax in fig.axes)
    plt.close(fig)


def test_transfer_archive_current_source_no_results_or_environments(tmp_path):
    import tarfile
    import hashlib
    import nbformat
    root = tmp_path / "workspace"
    root.mkdir()
    for name in ("src/new.py", "cpp/CMakeLists.txt", "results/old.json", "build/binary", ".conda-env/private.txt"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("working-tree fixture")
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1", execution_count=1,
        outputs=[nbformat.v4.new_output("stream", name="stdout", text="old result")])])
    nbpath = root / "notebooks/01_old.ipynb"
    nbpath.parent.mkdir()
    nbformat.write(nb, nbpath)
    original = nbpath.read_bytes()
    archive_path = tmp_path / "transfer.tar.gz"
    module = load_tool("package_linux_source.py")
    manifest = module.create_package(root, archive_path)
    assert nbpath.read_bytes() == original
    assert {f["path"] for f in manifest["files"]} == {"src/new.py", "cpp/CMakeLists.txt", "notebooks/01_old.ipynb"}
    with tarfile.open(archive_path) as archive:
        for entry in manifest["files"]:
            data = archive.extractfile("droplet-shadow-imaging/" + entry["path"]).read()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            if entry["path"].endswith(".ipynb"):
                assert json.loads(data)["cells"][0]["outputs"] == []
    with pytest.raises(FileExistsError):
        module.create_package(root, archive_path)


def test_notebook_06_valid_and_contains_no_transport_calls():
    import ast
    import nbformat
    notebook = nbformat.read(ROOT / "notebooks/06_field_scale_comparison.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                assert name not in {"simulate", "run_case", "run_geant4", "trace_rays", "main"}
