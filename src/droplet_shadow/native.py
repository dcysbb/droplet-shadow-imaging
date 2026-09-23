"""Python ↔ Geant4 的唯一数据桥梁。

Python 侧统一 SI；C++/Geant4 侧读入 mm、MeV、V/mm。这里导出
``source.csv``（每个初级电子）、``field.csv``（球对称径向场）和
``multipoles.csv``（非对称系数），运行本地可执行文件，再把
``hits/exits/deposition`` CSV 转回 SI 的 NumPy 数组。
除这个模块外，不应在 Python 模型中使用 Geant4 的单位系统。
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .config import SimulationConfig
from .constants import C, E_CHARGE, M_E
from .fields import RadialShells, SumField, SurfaceMultipoles, TabulatedRadial, UniformVolume, build_field


def _components(field: object) -> list[object]:
    """递归展开 ``SumField``，方便径向场和球谐场分别导出。"""
    if isinstance(field, SumField):
        return [piece for child in field.fields for piece in _components(child)]
    return [field]


def export_field(config: SimulationConfig, path: Path, multipole_path: Path) -> None:
    """写入 Geant4 读取的场表；在薄壳两侧额外采样以保留跳变。

径向表的三列分别是 ``radius_mm,potential_v,field_v_mm``；
球谐表另存，以免把具有角向结构的场错误地压缩到一维径向表。
"""
    field = build_field(config)
    R = config.geometry.radius_m
    inner = np.linspace(0, R, 550)
    extra: list[float] = []
    for item in _components(field):
        if isinstance(item, RadialShells):
            for radius in item.shell_radii_m:
                delta = min(1e-11, max(1e-13, (R-radius) / 20))
                # 壳前/壳上/壳后样本帮助 Geant4 不跨过不连续薄层。
                extra += [radius - delta, radius, radius + delta]
        elif isinstance(item, TabulatedRadial):
            extra.extend(item.radii_m.tolist())
    radii = np.unique(np.clip(np.r_[inner, extra], 0, R))
    points = np.zeros((len(radii), 3))
    points[:, 0] = radii
    radial_fields = [piece for piece in _components(field) if not isinstance(piece, SurfaceMultipoles)]
    radial = SumField(*radial_fields)
    phi = radial.potential(points)
    ex = radial.electric_field(points)[:, 0]
    if len(ex) > 1:
        ex[0] = 0.0
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["radius_mm", "potential_v", "field_v_mm"])
        writer.writerows(zip(radii * 1e3, phi, ex / 1e3))
    with multipole_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ell", "m", "coefficient_real_v", "coefficient_imag_v"])
        for item in _components(field):
            if isinstance(item, SurfaceMultipoles):
                for (ell, m), coefficient in item.coefficients.items():
                    if m >= 0:
                        # 实场的 -m 项由复共轭对称性恢复，CSV 只需 m≥0。
                        scaled = coefficient / R ** (ell + 1)
                        writer.writerow([ell, m, scaled.real, scaled.imag])


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


def run_geant4(config: SimulationConfig, phase: np.ndarray, momentum: np.ndarray,
               workdir: Path) -> dict[str, np.ndarray]:
    """执行完整 Geant4 输运并返回统一格式的落点、出滴和沉积记录。

临时 CSV 和日志保留在 ``workdir``，便于追查运行失败或复核单位。
``hits`` 包含到达屏幕的初级/次级电子；``exits`` 是所有穿滴后离开
水体的粒子，不受有限探测器接收孔径选择偏差影响。
"""
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    source_file, field_file = workdir / "source.csv", workdir / "field.csv"
    multipoles, hits_file = workdir / "multipoles.csv", workdir / "hits.csv"
    export_source(config, phase, momentum, source_file)
    export_field(config, field_file, multipoles)
    layers_nm = [config.charge.double_layer_thickness_nm
                 if config.charge.model == "neutral_double_layer" or
                    config.charge.double_layer_q_e else 0.0,
                 100.0 if config.charge.model == "poisson_boltzmann" else 0.0,
                 config.charge.dipole_thickness_nm
                 if config.charge.dipole_potential_v else 0.0]
    active_layers = [value for value in layers_nm if value > 0]
    # 单独的薄层步长不是纯性能参数：若一步跨过势垒，静电能量会失守恒。
    layer_nm = max(active_layers, default=0.0)
    layer_step_nm = (max(0.03, min(1.0, min(active_layers) / 6.0)) *
                     config.run.geant4_layer_step_scale) if active_layers else 0.0
    command = [str(geant4_executable(config)), str(source_file), str(field_file),
               str(multipoles), str(hits_file),
               str(config.geometry.radius_um * 1e-3),
               str(config.geometry.l1_mm), str(config.geometry.l2_mm),
               str(config.geometry.epsilon_water), config.run.droplet_material,
               str(config.run.seed), str(layer_nm), str(layer_step_nm),
               str(config.run.geant4_world_step_mm),
               str(config.run.geant4_far_step_mm),
               str(config.run.geant4_near_step_um),
               str(config.run.geant4_core_step_um),
               str(config.run.geant4_cut_um)]
    environment = os.environ.copy()
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
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=workdir, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"Geant4 exited {completed.returncode}; see {log_path}\n{tail}")
    rows = np.genfromtxt(hits_file, delimiter=",", names=True)
    # genfromtxt 在单行和多行文件会返回不同维度，atleast_1d 统一接口。
    rows = np.atleast_1d(rows)
    deposited = np.atleast_1d(np.genfromtxt(workdir / "deposition.csv", delimiter=",", names=True))["deposited_e"]
    exits = np.atleast_1d(np.genfromtxt(workdir / "exits.csv", delimiter=",", names=True))
    exit_data = {"exit_event_id": exits["event_id"].astype(int),
                 "exit_parent_id": exits["parent_id"].astype(int),
                 "exit_energy_mev": exits["energy_MeV"],
                 "exit_z_m": exits["z_mm"] * 1e-3,
                 "exit_ux": exits["ux"], "exit_uy": exits["uy"],
                 "exit_uz": exits["uz"]}
    if len(rows) == 0:
        return {"x_m": np.array([]), "y_m": np.array([]), "energy_mev": np.array([]),
                "entered": np.array([], dtype=bool), "blocked": np.array([], dtype=bool),
                "primary": np.array([], dtype=bool), "event_id": np.array([], dtype=int),
                "deposited_e_by_event": deposited, **exit_data}
    return {"x_m": rows["x_mm"] * 1e-3, "y_m": rows["y_mm"] * 1e-3,
            "energy_mev": rows["energy_MeV"], "entered": rows["entered"] > 0,
            "blocked": np.zeros(len(rows), dtype=bool), "primary": rows["parent_id"] == 0,
            "event_id": rows["event_id"].astype(int), "deposited_e_by_event": deposited,
            **exit_data}
