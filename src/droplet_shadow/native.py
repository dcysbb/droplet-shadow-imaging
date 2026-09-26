"""Python ↔ Geant4 的唯一数据桥梁。

Python 侧统一 SI；C++/Geant4 侧读入 mm、MeV、V/mm。这里导出
``source.csv``（每个初级电子）和 ``field.csv``（球对称径向场），运行本地可执行文件，再把
``hits/exits/deposition`` CSV 转回 SI 的 NumPy 数组。
除这个模块外，不应在 Python 模型中使用 Geant4 的单位系统。
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

from .config import SimulationConfig
from .constants import C, E_CHARGE, M_E
from .fields import RadialShells, SumField, TabulatedRadial, UniformVolume, build_field
from .progress import SimulationProgress


def _components(field: object) -> list[object]:
    """展开球对称场组合，收集全部电荷壳面、介电界面和 PB 网格。"""
    if isinstance(field, SumField):
        return [piece for child in field.fields for piece in _components(child)]
    return [field]


def export_field(config: SimulationConfig, path: Path) -> None:
    """写入 Geant4 径向场表，重复边界半径分别保存内、外侧场。

径向表的三列分别是 ``radius_mm,potential_v,field_v_mm``；
相同半径的两行按“内侧、外侧”排序。C++ 在区间内对电势作 Hermite
插值并取负导数，因此不会把球面跳变平滑成虚假的过渡场。
"""
    field = build_field(config)
    R = config.geometry.radius_m
    inner = np.linspace(0, R, 550)
    extra: list[float] = []
    boundaries = {R}
    for item in _components(field):
        if isinstance(item, RadialShells):
            boundaries.update(item.shell_radii_m)
        elif isinstance(item, TabulatedRadial):
            extra.extend(item.radii_m.tolist())
        if isinstance(item, (RadialShells, UniformVolume)) and item.layer_inner_m is not None:
            boundaries.add(item.layer_inner_m)
    # 每个界面区间都解析采样；不依赖毫米传播网格恰好命中纳米薄层。
    sorted_edges = sorted(boundaries)
    for low, high in zip(sorted_edges[:-1], sorted_edges[1:]):
        extra.extend(np.linspace(low, high, 65))
    radii = np.unique(np.r_[inner, extra, sorted_edges])
    table_r, evaluate_r = [], []
    for radius in radii:
        if radius in boundaries:
            table_r.append(radius)
            evaluate_r.append(np.nextafter(radius, 0.0))
        table_r.append(radius)
        evaluate_r.append(radius)
    points = np.zeros((len(table_r), 3))
    points[:, 0] = table_r
    phi = field.potential(points)
    points[:, 0] = evaluate_r
    ex = field.electric_field(points)[:, 0]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["radius_mm", "potential_v", "field_v_mm"])
        writer.writerows(zip(np.asarray(table_r) * 1e3, phi, ex / 1e3))


def export_source(config: SimulationConfig, phase: np.ndarray, momentum: np.ndarray,
                  path: Path) -> None:
    """把 SI 相空间转换成每行一个初级电子的 mm/斜率/MeV CSV。"""
    p2 = np.sum(momentum**2, axis=1)
    energy = (np.sqrt(M_E**2 * C**4 + p2 * C**2) - M_E * C**2) / (1e6 * E_CHARGE)
    source = np.column_stack((phase[:, :2] * 1e3,
                              momentum[:, :2] / momentum[:, 2, None], energy))
    np.savetxt(path, source, delimiter=",", header="x_mm,y_mm,xprime,yprime,energy_MeV", comments="")


def geant4_executable(config: SimulationConfig) -> Path:
    """取得用户指定或项目本机构建的 ``droplet_g4`` 可执行文件。"""
    value = config.run.geant4_executable or str(Path(__file__).resolve().parents[2] / "build" / "droplet_g4")
    executable = Path(value)
    if not executable.is_file():
        raise FileNotFoundError(f"Geant4 executable absent: {executable}. Build it with CMake first.")
    return executable


def resolve_threads(requested: int, events: int, multithreaded: bool) -> int:
    """解析一次运行的资源预算；不在参数扫描外层再开线程。

自动模式尊重可用 CPU affinity（若平台提供），留一核且最多八线程。
显式请求不设八线程上限，但没有必要使用比初级事件更多的 worker。
"""
    if type(requested) is not int or requested < 0 or events < 1:
        raise ValueError("Invalid thread or event count")
    if not multithreaded:
        if requested > 1:
            raise RuntimeError("Geant4 build lacks multithreading; use geant4_threads: 1")
        if requested == 0:
            warnings.warn("Geant4 lacks multithreading; auto mode falls back to serial", RuntimeWarning)
        return 1
    if requested:
        return min(requested, events)
    try:
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    except OSError:
        cpus = os.cpu_count()
    return min(events, 8, max(1, (cpus or 1) - 1))


def native_capabilities(executable: Path) -> dict:
    """旧二进制不会理解新分片协议；运行大规模模拟前立即提示重编译。"""
    try:
        result = subprocess.run([str(executable), "--capabilities"], capture_output=True,
                                text=True, timeout=30, check=True)
        capabilities = json.loads(result.stdout)
        if capabilities.get("protocol") != 2 or type(capabilities.get("multithreaded")) is not bool:
            raise ValueError("Unsupported protocol")
        return capabilities
    except (subprocess.SubprocessError, ValueError, AttributeError) as exc:
        raise RuntimeError("Incompatible Geant4 executable. Rebuild: cmake --build build") from exc


# 显式整数字段避免事件编号经过不必要的浮点转换，空表/单行也统一为一维。
CSV_SCHEMAS = {
    "hits": [("event_id", "i8"), ("track_id", "i8"), ("parent_id", "i8"),
             *[(name, "f8") for name in ("x_mm", "y_mm", "energy_MeV", "ux", "uy")],
             ("entered", "i8")],
    "exits": [("event_id", "i8"), ("track_id", "i8"), ("parent_id", "i8"),
              *[(name, "f8") for name in ("x_mm", "y_mm", "z_mm", "energy_MeV", "ux", "uy", "uz")]],
    "deposition": [("event_id", "i8"), ("deposited_e", "f8")],
}


def _read_csv(path: Path, kind: str) -> np.ndarray:
    dtype = np.dtype(CSV_SCHEMAS[kind])
    with path.open() as stream:
        if stream.readline().strip() != ",".join(dtype.names):
            raise RuntimeError(f"Invalid {kind} CSV header: {path}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
            rows = np.loadtxt(stream, delimiter=",", dtype=dtype, ndmin=1)
    if any(not np.all(np.isfinite(rows[name])) for name in dtype.names):
        raise RuntimeError(f"Non-finite native output: {path}")
    return rows


def merge_worker_outputs(directory: Path, destination: Path, events: int,
                         metadata: dict) -> dict[str, np.ndarray]:
    """验证后合并线程分片；不得按完成顺序解释 event_id。

同一轨迹可以多次穿出水滴，因此 exits 只稳定排序，绝不去重。
分片位于每次运行独有的目录，失败时保留，旧运行分片不会被误合并。
"""
    workers = metadata["workers"]
    if len(workers) != metadata["actual_threads"] or len({w["id"] for w in workers}) != len(workers):
        raise RuntimeError("Duplicate or missing workers")
    pieces = {kind: [] for kind in CSV_SCHEMAS}
    for worker in workers:
        current = {kind: _read_csv(directory / f"worker_{worker['id']}_{kind}.csv", kind)
                   for kind in CSV_SCHEMAS}
        ids = current["deposition"]["event_id"]
        if len(ids) != worker["events"] or len(np.unique(ids)) != len(ids):
            raise RuntimeError("Worker completed-event count mismatch")
        for kind in ("hits", "exits"):
            if not np.all(np.isin(current[kind]["event_id"], ids)):
                raise RuntimeError("Hit/exit does not belong to a completed worker event")
        for kind in pieces:
            pieces[kind].append(current[kind])
    merged = {}
    for kind, shards in pieces.items():
        rows = np.concatenate(shards) if shards else np.empty(0, dtype=CSV_SCHEMAS[kind])
        keys = (rows["track_id"], rows["event_id"]) if kind != "deposition" else (rows["event_id"],)
        merged[kind] = rows[np.lexsort(keys)]
    if not np.array_equal(merged["deposition"]["event_id"], np.arange(events)):
        raise RuntimeError("Missing, duplicate or out-of-range primary event IDs")
    # 所有完整性检查通过之后才生成标准文件；每个文件原子替换。
    for kind, rows in merged.items():
        temporary = destination / f"{kind}.csv.tmp"
        formats = ["%d" if rows.dtype[name].kind == "i" else "%.15g" for name in rows.dtype.names]
        np.savetxt(temporary, rows, delimiter=",", header=",".join(rows.dtype.names),
                   comments="", fmt=formats)
        temporary.replace(destination / f"{kind}.csv")
    return merged


def _read_progress(path: Path, progress: SimulationProgress) -> None:
    """只读原子快照。尚未生成/读取失败不影响输运，最终仍靠事件表验收。"""
    try:
        snapshot = json.loads(path.read_text())
        done = snapshot["completed"]
        if snapshot["total"] != progress.total or type(done) is not int or not 0 <= done <= progress.total:
            return
        if snapshot["phase"] == "transport":
            progress.stage("Geant4 输运")
            progress.update(done)
    except (OSError, ValueError, KeyError, TypeError):
        pass


def _run_native_process(command, workdir, environment, log_path, progress_path, progress):
    """不消费 Geant4 日志管道：全部日志照常保存，仅轮询小型进度快照。

Ctrl+C/Notebook 中断时终止并回收子进程，不能让隐藏的 Geant4 作业继续占满 CPU。
"""
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=workdir, env=environment,
                              stdout=log, stderr=subprocess.STDOUT) as process:
            try:
                while True:
                    try:
                        code = process.wait(timeout=.2)
                        if progress.enabled:
                            _read_progress(progress_path, progress)
                        return code
                    except subprocess.TimeoutExpired:
                        if progress.enabled:
                            _read_progress(progress_path, progress)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def run_geant4(config: SimulationConfig, phase: np.ndarray, momentum: np.ndarray,
               workdir: Path, runtime: dict | None = None,
               progress: SimulationProgress | None = None) -> dict[str, np.ndarray]:
    """执行完整 Geant4 输运并返回统一格式的落点、出滴和沉积记录。

临时 CSV 和日志保留在 ``workdir``，便于追查运行失败或复核单位。
``hits`` 包含到达屏幕的初级/次级电子；``exits`` 是所有穿滴后离开
水体的粒子，不受有限探测器接收孔径选择偏差影响。
"""
    if progress is None:
        with SimulationProgress(len(phase), config.run.show_progress) as display:
            return run_geant4(config, phase, momentum, workdir, runtime, display)
    progress.stage("准备 Geant4 输入")
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    executable = geant4_executable(config)
    capabilities = native_capabilities(executable)
    if progress.enabled and not capabilities.get("progress"):
        raise RuntimeError("Geant4 progress requires a rebuild: cmake --build build")
    threads = resolve_threads(config.run.geant4_threads, len(phase), capabilities["multithreaded"])
    shards = Path(tempfile.mkdtemp(prefix="workers_", dir=workdir))
    source_file, field_file = workdir / "source.csv", workdir / "field.csv"
    export_source(config, phase, momentum, source_file)
    export_field(config, field_file)
    layers_nm = [config.charge.double_layer_thickness_nm
                 if config.charge.model == "neutral_double_layer" or
                    config.charge.double_layer_q_e else 0.0,
                 100.0 if config.charge.model == "poisson_boltzmann" else 0.0,
                 config.charge.dipole_thickness_nm
                 if config.charge.dipole_potential_v else 0.0]
    active_layers = [value for value in layers_nm if value > 0]
    # 单独的薄层步长不是纯性能参数：若一步跨过势垒，静电能量会失守恒。
    layer_nm = max(active_layers, default=0.0)
    layer_step_nm = (min(1.0, min(active_layers) / 6.0) *
                     config.run.geant4_layer_step_scale) if active_layers else 0.0
    command = [str(executable), str(source_file), str(field_file), str(shards),
               str(config.geometry.radius_um * 1e-3),
               str(config.geometry.l1_mm), str(config.geometry.l2_mm),
               config.run.droplet_material,
               str(config.run.seed), str(layer_nm), str(layer_step_nm),
               str(config.run.geant4_world_step_mm),
               str(config.run.geant4_far_step_mm),
               str(config.run.geant4_near_step_um),
               str(config.run.geant4_core_step_um),
               str(config.run.geant4_cut_um), str(threads)]
    environment = os.environ.copy()
    environment["DROPLET_SHOW_PROGRESS"] = "1" if progress.enabled else "0"
    # Geant4 电磁物理过程需要官方数据集；从当前环境的 geant4-config 定位。
    candidate = shutil.which("geant4-config")
    config_command = Path(candidate) if candidate else Path(sys.prefix) / "bin" / "geant4-config"
    if config_command.is_file():
        datasets = subprocess.check_output([str(config_command), "--datasets"], text=True)
        for line in datasets.splitlines():
            parts = line.split(maxsplit=2)
            if len(parts) == 3:
                directory = Path(parts[2])
                if not directory.is_dir() and directory.name.startswith("G4"):
                    directory = directory.with_name(directory.name[2:])
                if directory.is_dir():
                    environment[parts[1]] = str(directory)
    log_path = workdir / "geant4.log"
    prepared = time.perf_counter()
    progress.stage("初始化 Geant4")
    returncode = _run_native_process(command, workdir, environment, log_path,
                                     shards / "progress.json", progress)
    if returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"Geant4 exited {returncode}; see {log_path}\n{tail}")
    transported = time.perf_counter()
    progress.stage("合并与检查事件记录")
    metadata = json.loads((shards / "transport.json").read_text())
    if metadata.get("protocol") != 2 or metadata.get("actual_threads") != threads:
        raise RuntimeError("Native runtime protocol/thread count mismatch")
    merged = merge_worker_outputs(shards, workdir, len(phase), metadata)
    progress.update(len(phase))
    rows, exits = merged["hits"], merged["exits"]
    deposited = merged["deposition"]["deposited_e"]
    metadata.update(requested_threads=config.run.geant4_threads, shard_directory=shards.name,
                    input_preparation_s=prepared-started,
                    native_process_s=transported-prepared,
                    merge_read_s=time.perf_counter()-transported)
    (workdir / "transport.json").write_text(json.dumps(metadata, indent=2))
    if runtime is not None:
        runtime.update(metadata)
    exit_data = {"exit_event_id": exits["event_id"].astype(int),
                 "exit_track_id": exits["track_id"].astype(int),
                 "exit_parent_id": exits["parent_id"].astype(int),
                 "exit_energy_mev": exits["energy_MeV"],
                 "exit_z_m": exits["z_mm"] * 1e-3,
                 "exit_ux": exits["ux"], "exit_uy": exits["uy"],
                 "exit_uz": exits["uz"]}
    if len(rows) == 0:
        return {"x_m": np.array([]), "y_m": np.array([]), "energy_mev": np.array([]),
                "entered": np.array([], dtype=bool), "blocked": np.array([], dtype=bool),
                "primary": np.array([], dtype=bool), "event_id": np.array([], dtype=int),
                "track_id": np.array([], dtype=int), "parent_id": np.array([], dtype=int),
                "deposited_e_by_event": deposited, **exit_data}
    return {"x_m": rows["x_mm"] * 1e-3, "y_m": rows["y_mm"] * 1e-3,
            "energy_mev": rows["energy_MeV"], "entered": rows["entered"] > 0,
            "blocked": np.zeros(len(rows), dtype=bool), "primary": rows["parent_id"] == 0,
            "event_id": rows["event_id"].astype(int), "deposited_e_by_event": deposited,
            "track_id": rows["track_id"].astype(int), "parent_id": rows["parent_id"].astype(int),
            **exit_data}
