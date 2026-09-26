"""事件并行的配置、分片完整性和端到端测试；不以速度决定 CI 成败。"""

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from droplet_shadow.config import Run, SimulationConfig
from droplet_shadow.detector import image_shape, preflight_image
from droplet_shadow.native import (CSV_SCHEMAS, merge_worker_outputs, native_capabilities,
                                   resolve_threads, run_geant4)
from droplet_shadow.source import sample_source


@pytest.mark.parametrize("value", [-1, 1.2, True, "4", None])
def test_invalid_threads(value):
    with pytest.raises(ValueError, match="geant4_threads"):
        Run(geant4_threads=value)


def test_thread_budget(monkeypatch):
    monkeypatch.delattr("os.sched_getaffinity", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 10)
    assert resolve_threads(0, 100, True) == 8
    assert resolve_threads(0, 2, True) == 2
    assert resolve_threads(1, 100, True) == 1
    assert resolve_threads(12, 100, True) == 12
    assert resolve_threads(12, 3, True) == 3
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert resolve_threads(0, 100, True) == 1
    with pytest.warns(RuntimeWarning, match="falls back"):
        assert resolve_threads(0, 100, False) == 1
    with pytest.raises(RuntimeError, match="lacks multithreading"):
        resolve_threads(4, 100, False)


def test_old_native_protocol(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "old binary"))
    with pytest.raises(RuntimeError, match="cmake --build build"):
        native_capabilities(Path("not-executed"))


def test_large_raster_warns_without_allocating():
    cfg = SimulationConfig().with_updates(run={"raster_width_mm": 750})
    assert image_shape(cfg) == (30000, 30000)
    with pytest.warns(RuntimeWarning, match="7.20 GB"):
        preflight_image(cfg)
    assert image_shape(cfg.with_updates(run={"raster_width_mm": 40})) == (1600, 1600)


def _shard(root, worker, kind, rows):
    names = [name for name, _ in CSV_SCHEMAS[kind]]
    (root / f"worker_{worker}_{kind}.csv").write_text(
        ",".join(names) + "\n" + "".join(",".join(map(str, row)) + "\n" for row in rows))


def _two_workers(root):
    # 线程完成顺序不等于事件顺序。第二个 worker 无探测器落点。
    for worker in (0, 1):
        for kind in ("hits", "exits"):
            _shard(root, worker, kind, [])
    _shard(root, 0, "deposition", [(2, -1), (0, 0)])
    _shard(root, 1, "deposition", [(1, 1)])
    _shard(root, 0, "hits", [(2, 2, 1, 0, 0, 1, 0, 0, 1), (0, 1, 0, 0, 0, 3, 0, 0, 0)])
    # 同一轨迹两次出滴，必须保留原先发生顺序，不能按能量再排序或去重。
    _shard(root, 0, "exits", [(2, 1, 0, 0, 0, .05, 3, 0, 0, 1),
                              (2, 1, 0, 0, 0, -.05, 2, 0, 0, -1)])
    return {"actual_threads": 2, "workers": [{"id": 0, "events": 2}, {"id": 1, "events": 1}]}


def test_merge_event_ids_secondary_and_repeated_exit(tmp_path):
    metadata = _two_workers(tmp_path)
    merged = merge_worker_outputs(tmp_path, tmp_path, 3, metadata)
    np.testing.assert_array_equal(merged["deposition"]["deposited_e"], [0, 1, -1])
    np.testing.assert_array_equal(merged["hits"]["event_id"], [0, 2])
    np.testing.assert_array_equal(merged["hits"]["parent_id"], [0, 1])
    np.testing.assert_array_equal(merged["exits"]["energy_MeV"], [3, 2])


@pytest.mark.parametrize("bad_ids", [[0, 0], [0, 3], [0]])
def test_missing_duplicate_or_invalid_event_fails(tmp_path, bad_ids):
    metadata = _two_workers(tmp_path)
    _shard(tmp_path, 0, "deposition", [(event, 0) for event in bad_ids])
    with pytest.raises(RuntimeError):
        merge_worker_outputs(tmp_path, tmp_path, 3, metadata)
    assert not (tmp_path / "hits.csv").exists()
    assert (tmp_path / "worker_0_hits.csv").exists()


EXECUTABLE = Path(__file__).resolve().parents[1] / "build" / "droplet_g4"


@pytest.mark.skipif(not EXECUTABLE.exists(), reason="Geant4 not built")
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_native_thread_accounting_and_reproducibility(tmp_path, threads):
    cfg = SimulationConfig().with_updates(
        source={"crossover_fwhm_um": 3, "divergence_rms_mrad": 1},
        run={"engine": "geant4", "n_simulated": 48, "geant4_threads": threads})
    phase, momentum = sample_source(cfg.source, 48, np.random.default_rng(7))
    runs = [run_geant4(cfg, phase, momentum, tmp_path / str(i)) for i in range(2)]
    for key in runs[0]:
        np.testing.assert_array_equal(runs[0][key], runs[1][key], err_msg=key)
    runtime = json.loads((tmp_path / "0" / "transport.json").read_text())
    assert runtime["actual_threads"] == threads
    assert sum(worker["events"] for worker in runtime["workers"]) == 48
    assert len(runs[0]["deposited_e_by_event"]) == 48
    assert np.all((runs[0]["event_id"] >= 0) & (runs[0]["event_id"] < 48))


@pytest.mark.skipif(not EXECUTABLE.exists(), reason="Geant4 not built")
def test_native_no_hits_and_single_event(tmp_path):
    cfg = SimulationConfig().with_updates(charge={"q_e": 0},
        run={"engine": "geant4", "n_simulated": 1, "geant4_threads": 4,
             "droplet_material": "vacuum"})
    phase, momentum = sample_source(cfg.source, 1, np.random.default_rng(1))
    phase[0, :2] = [.09, 0]  # 起点在 world 内、屏幕外，方向沿 z。
    momentum[0, :2] = 0
    runtime = {}
    hits = run_geant4(cfg, phase, momentum, tmp_path, runtime)
    assert runtime["actual_threads"] == 1
    assert len(hits["x_m"]) == 0
    np.testing.assert_array_equal(hits["deposited_e_by_event"], [0])


@pytest.mark.skipif(not EXECUTABLE.exists(), reason="Geant4 not built")
def test_native_event_source_mapping(tmp_path):
    """不同事件给不同位置/斜率；合并后逐事件核对几何投影而非只看总数。"""
    cfg = SimulationConfig().with_updates(charge={"q_e": 0},
        run={"engine": "geant4", "n_simulated": 17, "geant4_threads": 4,
             "droplet_material": "vacuum"})
    phase, momentum = sample_source(cfg.source, 17, np.random.default_rng(9))
    hits = run_geant4(cfg, phase, momentum, tmp_path)
    np.testing.assert_array_equal(hits["event_id"], np.arange(17))
    # 记录面是厚 1 μm 的真空薄片，入射面位于 L2-0.5 μm。
    length = (cfg.geometry.l1_mm + cfg.geometry.l2_mm) * 1e-3 - .5e-6
    expected = phase[:, :2] + length * momentum[:, :2] / momentum[:, 2, None]
    np.testing.assert_allclose(hits["x_m"], expected[:, 0], atol=1e-10)
    np.testing.assert_allclose(hits["y_m"], expected[:, 1], atol=1e-10)
