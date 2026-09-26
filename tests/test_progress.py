"""进度只影响观测，不改变随机流、输运结果或失败判断。"""

import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from droplet_shadow.config import Run, SimulationConfig
from droplet_shadow.native import _read_progress, _run_native_process, run_geant4
from droplet_shadow.progress import SimulationProgress
from droplet_shadow.source import sample_source


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_progress_boolean_validation(value):
    with pytest.raises(ValueError, match="show_progress"):
        Run(show_progress=value)


def test_progress_monotonic_eta_and_no_false_success(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("droplet_shadow.progress.time.monotonic", lambda: clock[0])
    stream = io.StringIO()
    with pytest.raises(RuntimeError):
        with SimulationProgress(10, stream=stream) as p:
            p.stage("Geant4 输运")
            clock[0] = 5.0
            p.update(5)
            p.update(2)
            assert p.done == 5
            raise RuntimeError("failed transport")
    assert "50.0%" in stream.getvalue()
    assert "预计剩余 5s" in stream.getvalue()
    assert "失败" in stream.getvalue()
    assert "完成 [" not in stream.getvalue()
    assert "100.0%" not in stream.getvalue()


def test_disabled_and_redirected_throttle(monkeypatch):
    stream = io.StringIO()
    with SimulationProgress(10, enabled=False, stream=stream) as p:
        p.stage("Geant4 输运")
        p.update(10)
    assert stream.getvalue() == ""
    monkeypatch.setattr("droplet_shadow.progress.time.monotonic", lambda: 0.0)
    with SimulationProgress(100, stream=stream) as p:
        p.stage("Geant4 输运")
        for i in range(101):
            p.update(i)
    assert len(stream.getvalue().splitlines()) == 2  # 阶段起点和结束，不逐事件刷屏。


def test_terminal_single_line():
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    stream = Terminal()
    with SimulationProgress(1, stream=stream) as p:
        p.stage("Geant4 输运")
        p.update(1)
    assert "\r\033[2K" in stream.getvalue()
    assert stream.getvalue().endswith("\n")
    assert stream.getvalue().count("\n") == 1


def test_notebook_reuses_display(monkeypatch):
    display_module = pytest.importorskip("IPython.display")
    rendered = []
    class Handle:
        def update(self, value):
            rendered.append(value.data)
    calls = []
    def display(value, display_id):
        calls.append(display_id)
        rendered.append(value.data)
        return Handle()
    monkeypatch.setattr(display_module, "display", display)
    with SimulationProgress(2, stream=io.StringIO()) as p:
        p.notebook = True
        p.stage("Geant4 输运")
        p.update(2)
        p.stage("保存结果")
    assert calls == [True]
    assert "<progress" in rendered[0]
    assert "完成" in rendered[-1]


def test_snapshot_and_live_process(tmp_path):
    path = tmp_path / "progress.json"
    p = SimulationProgress(5, stream=io.StringIO())
    _read_progress(path, p)  # 初始化尚未写文件不应报错。
    path.write_text("incomplete")
    _read_progress(path, p)
    assert p.done == 0
    # 小型独立进程模拟原子快照；验证子进程仍在运行时能取得中间计数。
    observed = []
    original_update = p.update
    def update(n):
        observed.append(n)
        original_update(n)
    p.update = update
    program = (
        "import json,time; from pathlib import Path; "
        "p=Path('progress.json'); t=Path('progress.tmp'); "
        "print('native log remains available',flush=True)\n"
        "for n in range(1,6):\n"
        " t.write_text(json.dumps(dict(completed=n,total=5,phase='transport')))\n"
        " t.replace(p)\n"
        " time.sleep(.2)\n"
    )
    code = _run_native_process([sys.executable, "-c", program], tmp_path, None,
                               tmp_path / "geant4.log", path, p)
    assert code == 0 and p.done == 5
    assert any(0 < n < 5 for n in observed)
    assert observed == sorted(observed)
    assert "native log" in (tmp_path / "geant4.log").read_text()


def test_interrupt_terminates_child(monkeypatch, tmp_path):
    class Process:
        terminated = False
        waits = 0
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def poll(self): return None
        def terminate(self): self.terminated = True
        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt()
            return -15
    child = Process()
    monkeypatch.setattr("droplet_shadow.native.subprocess.Popen", lambda *a, **k: child)
    with pytest.raises(KeyboardInterrupt):
        _run_native_process([], tmp_path, None, tmp_path / "geant4.log", tmp_path / "progress.json",
                            SimulationProgress(10, stream=io.StringIO()))
    assert child.terminated and child.waits == 2


@pytest.mark.skipif(not Path("build/droplet_g4").exists(), reason="Geant4 not built")
@pytest.mark.parametrize("threads", [1, 4])
def test_native_progress_does_not_change_results(tmp_path, threads):
    cfg = SimulationConfig().with_updates(run={"engine": "geant4", "n_simulated": 100,
                                               "geant4_threads": threads})
    phase, momentum = sample_source(cfg.source, 100, np.random.default_rng(10))
    visible = run_geant4(cfg, phase, momentum, tmp_path / "visible")
    hidden = run_geant4(cfg.with_updates(run={"show_progress": False}), phase, momentum,
                        tmp_path / "hidden")
    for key in visible:
        np.testing.assert_array_equal(visible[key], hidden[key], err_msg=key)
    snapshots = list((tmp_path / "visible").glob("workers_*/progress.json"))
    assert len(snapshots) == 1
    assert json.loads(snapshots[0].read_text())["completed"] == 100
    assert not list((tmp_path / "hidden").glob("workers_*/progress.json"))
