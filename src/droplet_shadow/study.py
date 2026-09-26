"""三类电场研究的固定矩阵、可审计清单与可恢复运行。

本模块导入、构造配置、读取清单都不会启动电子输运。只有显式调用
``run_case`` 才运行一次 Geant4。每次尝试使用新目录；完成标记在
配置、版本、事件和文件完整性检查之后写入，失败数据不覆盖、不删除。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from contextlib import contextmanager

import numpy as np
import scipy
import yaml

from .config import Charge, Detector, Geometry, Run, SimulationConfig, Source, load_config
from .detector import image_shape

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "examples" / "field_scale_study" / "manifest.json"
EXPOSURES = (100_000, 1_000_000, 10_000_000)
AVOGADRO = 6.02214076e23  # mol^-1，SI 精确定义值。


@dataclass(frozen=True)
class StudyCase:
    """清单元数据与可直接交给 simulate 的配置；元数据不塞入 YAML 模型字段。"""

    case_id: str
    group: str
    model: str
    config: SimulationConfig
    neutral_case: str
    absent_case: str
    note: str


def ion_count(concentration_umol_l: float, radius_m: float) -> float:
    """平均 μmol/L → 粒子总数；1 m³=1000 L，PB 允许连续平均粒子数。"""
    volume_l = 4 * math.pi * radius_m**3 / 3 * 1000
    return concentration_umol_l * 1e-6 * volume_l * AVOGADRO


def planned_cases() -> list[StudyCase]:
    """唯一矩阵定义：24 主组 + 6 模型补充 + 5 短探测距，全部 PSF=0。

每个模型从全新 Charge 对象构造，避免上一情景的离子/偶极参数残留。
固定发散角缩焦意味着几何发射度降低，不代表同一束流的纯透镜调节。
"""
    base = SimulationConfig(
        geometry=Geometry(radius_um=50, l1_mm=10, l2_mm=2000),
        source=Source(energy_mev=3, crossover_fwhm_um=10, divergence_rms_mrad=10),
        detector=Detector(psf_fwhm_um=0, pixel_um=25, diameter_mm=40),
        run=Run(engine="geant4", n_simulated=10_000_000, exposure_electrons=1_000_000,
                raster_width_mm=40, seed=20260923, geant4_threads=0, show_progress=True))
    models = {
        "absent": ({"q_e": 0}, "无滴、无场真空参照；不是中性水滴。"),
        "neutral": ({"q_e": 0}, "无电场水滴参照；保留水中散射和能损。"),
        "dipole_0p5nm": ({"q_e": 0, "dipole_potential_v": 1, "dipole_thickness_nm": .5},
                           "Hao 2022 启发的等效 1 V/0.5 nm 层；厚度/形状是假设，不是瞬时分子场。"),
        "q_plus_1e6": ({"q_e": 1e6}, "规定的均匀表面净电荷 +10^6 e；不是文献实测。"),
        "q_minus_1e6": ({"q_e": -1e6}, "规定的均匀表面净电荷 -10^6 e；不是文献实测。"),
        "dipole_1nm": ({"q_e": 0, "dipole_potential_v": 1, "dipole_thickness_nm": 1},
                         "相同等效势差 1 V、厚度改为 1 nm；隔离厚度变化的对照。"),
        "shell_10nm": ({"model": "neutral_double_layer", "q_e": 0,
                         "double_layer_q_e": 1e8, "double_layer_thickness_nm": 10},
                        "规定 ±10^8 e 同心双壳；总电荷抵消而非面密度相等，不是 PB 平衡解。"),
    }
    for label, concentration in (("pb_0p01uM", .01), ("pb_0p1uM", .1), ("pb_1uM", 1)):
        count = ion_count(concentration, base.geometry.radius_m)
        models[label] = ({"model": "poisson_boltzmann", "q_e": 0,
                          "surface_fixed_e": -count, "ion_positive_count": count},
                         f"球对称 PB；平均未配对正离子 {concentration:g} μmol/L；固定负表面电荷严格中和。")
    for label, q in (("q_plus_1e5", 1e5), ("q_minus_1e5", -1e5),
                     ("q_plus_1e7", 1e7), ("q_minus_1e7", -1e7)):
        models[label] = ({"q_e": q}, "净电荷大小/符号补充；规定分布，非实测电荷。")
    main = ("absent", "neutral", "dipole_0p5nm", "pb_0p01uM", "pb_0p1uM",
            "pb_1uM", "q_plus_1e6", "q_minus_1e6")
    cases = []

    def add(model, focus, distance, group):
        values, note = models[model]
        cfg = replace(base, charge=Charge(**values)).with_updates(
            geometry={"l2_mm": distance}, source={"crossover_fwhm_um": focus},
            run={"droplet_material": "vacuum" if model == "absent" else "water"})
        suffix = f"s{focus:g}_l{distance:g}"
        cases.append(StudyCase(f"{model}_{suffix}", group, model, cfg,
                               f"neutral_{suffix}", f"absent_{suffix}", note))

    for focus in (10, 5, 1):
        for model in main:
            add(model, focus, 2000, "main")
    for model in ("q_plus_1e5", "q_minus_1e5", "q_plus_1e7", "q_minus_1e7",
                  "dipole_1nm", "shell_10nm"):
        add(model, 10, 2000, "model_extra")
    for model in ("absent", "neutral", "dipole_0p5nm", "pb_0p1uM", "q_plus_1e6"):
        add(model, 10, 500, "distance_extra")
    return cases


def manifest_data(cases: list[StudyCase]) -> dict:
    """随源码交付的静态清单只声明 not_run；真实完成状态位于输出目录。"""
    return {"schema": 1, "description": "35 组球对称场/焦点对照；PSF=0；未运行",
            "exposure_electrons": list(EXPOSURES),
            "cases": [{"case_id": c.case_id, "group": c.group, "model": c.model,
                       "config": c.case_id + ".yaml", "neutral_case": c.neutral_case,
                       "absent_case": c.absent_case, "note": c.note, "status": "not_run"}
                      for c in cases]}


def load_study(path: Path = DEFAULT_MANIFEST) -> list[StudyCase]:
    """读取显式 YAML，不在执行时悄悄用默认值重建覆盖用户配置。"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1:
        raise ValueError("Unsupported study manifest schema")
    cases = []
    for item in data["cases"]:
        name = item["case_id"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError(f"Invalid case_id: {name!r}")
        config_path = path.parent / item["config"]
        if not config_path.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("Configuration must be inside the study directory")
        cases.append(StudyCase(name, item["group"], item["model"], load_config(config_path),
                               item["neutral_case"], item["absent_case"], item["note"]))
    ids = {c.case_id for c in cases}
    if len(ids) != len(cases):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if case.neutral_case not in ids or case.absent_case not in ids:
            raise ValueError(f"Missing reference for {case.case_id}")
    return cases


def config_from_dict(data: dict) -> SimulationConfig:
    """从保存的完整配置恢复对象；不需要目标机拥有原始 YAML 的绝对路径。"""
    classes = dict(geometry=Geometry, charge=Charge, source=Source, detector=Detector, run=Run)
    if set(data) != set(classes):
        raise ValueError("Saved result must contain all five configuration sections")
    return SimulationConfig(**{name: cls(**data[name]) for name, cls in classes.items()})


def sha256(path: Path) -> str:
    """流式计算大型 CSV/NPZ 哈希，不把数 GB 文件一次读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """替换小型状态文件，避免中断留下半个 JSON 被误读为完成。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def execution_fingerprint(config: SimulationConfig) -> dict:
    """仅真实运行时检查二进制；默认列清单不要求已安装 Geant4。"""
    from .simulation import version_fingerprint
    return {**version_fingerprint(config), "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": scipy.__version__, "yaml": yaml.__version__,
            "platform": platform.system(), "machine": platform.machine()}


REQUIRED_OUTPUTS = ("result.json", "config.yaml", "hits.npz", "images.npz", "shadow.png",
                    "shadow.svg", "native/source.csv", "native/field.csv", "native/hits.csv",
                    "native/exits.csv", "native/deposition.csv", "native/transport.json",
                    "native/geant4.log")


def validate_output(path: Path, config: SimulationConfig) -> None:
    """语义验收，允许零落点，但不允许缺失初级事件或错配配置。"""
    for name in REQUIRED_OUTPUTS:
        if not (path / name).is_file():
            raise RuntimeError(f"Missing output: {path / name}")
    report = json.loads((path / "result.json").read_text())
    if report["config"] != config.to_dict() or load_config(path / "config.yaml") != config:
        raise RuntimeError("Output configuration mismatch")
    metadata = json.loads((path / "native/transport.json").read_text())
    if sum(w["events"] for w in metadata["workers"]) != config.run.n_simulated:
        raise RuntimeError("Native completed-event count mismatch")
    with np.load(path / "hits.npz", allow_pickle=False) as hits:
        ids = hits["event_id"]
        if ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids >= config.run.n_simulated)):
            raise RuntimeError("Invalid hit event IDs")
        for key in ("x_m", "y_m", "energy_mev", "entered", "blocked", "primary", "track_id", "parent_id"):
            values = hits[key]
            if values.shape != ids.shape or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Invalid hit column {key}")
        deposition = hits["deposited_e_by_event"]
        if deposition.shape != (config.run.n_simulated,) or not np.all(np.isfinite(deposition)):
            raise RuntimeError("Invalid deposition-by-event array")
    with np.load(path / "images.npz", allow_pickle=False) as images:
        shape = image_shape(config)
        for key in ("ideal", "expected", "observed"):
            image = images[key]
            if image.shape != shape or not np.all(np.isfinite(image)):
                raise RuntimeError(f"Invalid image {key}")


def completed_path(case: StudyCase, output: Path, fingerprint: dict | None = None,
                   verify_hashes: bool = True) -> Path | None:
    """只返回经校验的完成尝试；None 表示无完成标记，损坏则明确报错。"""
    state_path = Path(output) / case.case_id / "state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    if state.get("status") != "complete":
        return None
    if state.get("config") != case.config.to_dict():
        raise RuntimeError(f"Changed configuration: {case.case_id}; use a new output root")
    if fingerprint is not None and state.get("fingerprint") != fingerprint:
        raise RuntimeError(f"Changed code/environment: {case.case_id}; use a new output root")
    attempt = Path(output) / case.case_id / state["attempt"]
    if not attempt.resolve().is_relative_to(state_path.parent.resolve()):
        raise RuntimeError("Invalid attempt path")
    hashes = state.get("sha256", {})
    if set(hashes) != set(REQUIRED_OUTPUTS):
        raise RuntimeError("Incomplete checksum manifest")
    for name in REQUIRED_OUTPUTS:
        if not (attempt / name).is_file() or (verify_hashes and sha256(attempt / name) != hashes[name]):
            raise RuntimeError(f"Output integrity check failed: {attempt / name}")
    return attempt


@contextmanager
def study_lock(output: Path):
    """防止两个调度器同时写同一研究目录。异常退出的残留锁需人工确认后移除。"""
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".study.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "host": platform.node()}, stream)
    try:
        yield
    finally:
        lock.unlink()


def resource_estimate(config: SimulationConfig) -> dict:
    """粗略资源预算而非保证；次级粒子数会影响实际占用，不修改物理配置。"""
    pixels = math.prod(image_shape(config))
    return {"estimated_peak_memory_bytes": config.run.n_simulated * 640 + pixels * 8 * 12,
            "estimated_disk_bytes": config.run.n_simulated * 512 + pixels * 8 * 4,
            "note": "情景估计，不是测量；次级粒子/CSV/压缩率会改变实际占用。"}


def run_case(case: StudyCase, output: Path, *, resume: bool = False) -> Path:
    """显式运行一组；调用者持有 study_lock，失败时保留独立 attempt。

本函数不自动调用三组 controls，因为研究清单已经共享匹配的无滴/中性
参照；也不调用环判别、焦散分类、电荷拟合或检出限分析。
"""
    from .simulation import simulate
    fingerprint = execution_fingerprint(case.config)
    folder = Path(output) / case.case_id
    if folder.exists():
        if not resume:
            raise FileExistsError(f"{folder} exists; use --resume or a new output root")
        cached = completed_path(case, output, fingerprint)
        if cached is not None:
            print(f"Verified cache: {case.case_id}", flush=True)
            return cached
    folder.mkdir(parents=True, exist_ok=True)
    # 空间门槛只保护下一组；全研究估计由列表输出供用户判断。
    needed = resource_estimate(case.config)["estimated_disk_bytes"]
    if shutil.disk_usage(folder).free < needed * 1.25:
        raise RuntimeError(f"Insufficient estimated disk space for {case.case_id}; no transport started")
    number = 1
    while (folder / f"attempt-{number:04d}").exists():
        number += 1
    attempt = folder / f"attempt-{number:04d}"
    attempt.mkdir()
    state = {"status": "running", "attempt": attempt.name, "config": case.config.to_dict(),
             "fingerprint": fingerprint, "started_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(folder / "state.json", state)
    (attempt / "config.yaml").write_text(yaml.safe_dump(case.config.to_dict(), sort_keys=False),
                                         encoding="utf-8")
    try:
        result = simulate(case.config, attempt)
        # 及时释放数百万条记录，避免下一组与上一组同时占据内存。
        del result
        validate_output(attempt, case.config)
        produced = json.loads((attempt / "result.json").read_text()).get("versions", {})
        for key in ("source_sha256", "geant4_executable_sha256"):
            if key in fingerprint and produced.get(key) != fingerprint[key]:
                raise RuntimeError("Code/binary changed during transport; refusing completion")
        state["sha256"] = {name: sha256(attempt / name) for name in REQUIRED_OUTPUTS}
        state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(attempt / "attempt_state.json", state)
        atomic_json(folder / "state.json", state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(attempt / "attempt_state.json", state)
        atomic_json(folder / "state.json", state)
        raise
    return attempt
