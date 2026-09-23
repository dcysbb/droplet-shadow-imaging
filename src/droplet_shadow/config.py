"""模拟配置的唯一入口：参数、单位和不允许的物理组合都在这里检查。

示例 YAML 按 ``geometry/charge/source/detector/run`` 分组；字段名的后缀
``_um/_nm/_mm/_mev`` 明确表示输入单位。电场和 Python 轨迹内部使用 SI，
所以调用方不应自行把 YAML 中的数字转换成米。所有配置对象都是冻结的：
参数扫描通过 ``with_updates`` 产生新对象，不会悄悄修改基准配置。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Geometry:
    """球形液滴与点投影几何；原点在液滴中心，束腰 z=-L1，屏幕 z=+L2。"""

    radius_um: float = 50.0
    l1_mm: float = 10.0
    l2_mm: float = 500.0
    temperature_k: float = 298.0
    epsilon_water: float = 78.0
    surface_tension_n_m: float = 0.072

    def __post_init__(self) -> None:
        # 表面张力仅用于 Rayleigh 极限，不决定本程序给定的电荷分布。
        if min(self.radius_um, self.l1_mm, self.l2_mm, self.temperature_k,
               self.epsilon_water, self.surface_tension_n_m) <= 0:
            raise ValueError("Geometry lengths, temperature, dielectric constant and tension must be positive")

    @property
    def radius_m(self) -> float:
        """电场公式所需的液滴半径（m）。"""
        return self.radius_um * 1e-6

    @property
    def magnification(self) -> float:
        """不含电场/散射时的几何放大率 M=(L1+L2)/L1。"""
        return (self.l1_mm + self.l2_mm) / self.l1_mm


@dataclass(frozen=True)
class Charge:
    """自由电荷与界面层情景，电荷数均以元电荷 e 为单位。

``q_e`` 是整个液滴的*净*电荷；中性双层的 ``double_layer_q_e``
表示内壳总电荷，外壳是相反的总电荷而非相反的面密度。
``dipole_potential_v`` 是连续介质等效势差，不是探针测得的瞬时分子场。
球谐比例表示固定/冻结的不对称表面电荷，不代表导体平衡态。
"""

    model: str = "surface"  # surface, volume, neutral_double_layer, poisson_boltzmann
    q_e: float = 1e6
    double_layer_q_e: float = 0.0
    double_layer_thickness_nm: float = 1.0
    dipole_potential_v: float = 0.0
    dipole_thickness_nm: float = 0.5
    dipole_epsilon: float = 2.0
    ion_positive_count: float = 0.0
    ion_negative_count: float = 0.0
    surface_fixed_e: float = 0.0
    dipole_fraction: float = 0.0
    quadrupole_fraction: float = 0.0
    patch_fraction: float = 0.0
    patch_width_deg: float = 30.0
    multipole_order: int = 8

    def __post_init__(self) -> None:
        # 先排除数值无效的参数，再排除需要重新自洽求解离子分布的组合。
        if self.model not in {"surface", "volume", "neutral_double_layer", "poisson_boltzmann"}:
            raise ValueError(f"Unknown charge model: {self.model}")
        if self.double_layer_thickness_nm <= 0 or self.dipole_thickness_nm <= 0 or self.dipole_epsilon <= 0:
            raise ValueError("Layer thicknesses and dipole-layer permittivity must be positive")
        if self.ion_positive_count < 0 or self.ion_negative_count < 0:
            raise ValueError("Ion counts must be non-negative")
        if not 0 <= self.multipole_order <= 16:
            raise ValueError("multipole_order must be 0..16")
        if self.patch_width_deg <= 0:
            raise ValueError("patch_width_deg must be positive")
        if self.model == "poisson_boltzmann" and (
            self.double_layer_q_e or self.dipole_potential_v or
            self.dipole_fraction or self.quadrupole_fraction or self.patch_fraction
        ):
            # PB 的移动离子会响应额外界面层；直接线性叠加会漏掉屏蔽。
            raise ValueError("PB plus another interface charge/polarization requires a coupled ion re-solve")
        if self.model != "poisson_boltzmann" and (
            self.ion_positive_count or self.ion_negative_count or self.surface_fixed_e
        ):
            raise ValueError("Ion counts and fixed PB surface charge require the poisson_boltzmann model")


@dataclass(frozen=True)
class Source:
    """有限束腰电子源；位置用 FWHM，角度和能散用单轴 RMS。

相空间相关系数允许束腰并非严格零相关；几何发射度由协方差计算，
而不是作为可独立随意设置的又一个源参数。
"""

    energy_mev: float = 3.0
    energy_spread_rms: float = 0.001
    crossover_fwhm_um: float = 10.0
    divergence_rms_mrad: float = 10.0
    x_xprime_correlation: float = 0.0
    y_yprime_correlation: float = 0.0

    def __post_init__(self) -> None:
        if self.energy_mev <= 0 or self.energy_spread_rms < 0 or self.crossover_fwhm_um <= 0 or self.divergence_rms_mrad <= 0:
            raise ValueError("Source energy and widths must be positive")
        if abs(self.x_xprime_correlation) >= 1 or abs(self.y_yprime_correlation) >= 1:
            raise ValueError("Source correlations must have magnitude below one")


@dataclass(frozen=True)
class Detector:
    """MCP/荧光屏和像素的情景参数；增益以任意信号单位计。"""

    psf_fwhm_um: float = 50.0
    pixel_um: float = 25.0
    diameter_mm: float = 40.0
    efficiency: float = 1.0
    gain_mean: float = 1.0
    gain_shape: float = 4.0
    background_counts_pixel: float = 0.0
    read_noise_rms: float = 0.0

    def __post_init__(self) -> None:
        if min(self.psf_fwhm_um, self.pixel_um, self.diameter_mm,
               self.gain_mean, self.gain_shape) <= 0:
            raise ValueError("Detector widths, size and gain parameters must be positive")
        if not 0 <= self.efficiency <= 1 or min(self.background_counts_pixel, self.read_noise_rms) < 0:
            raise ValueError("Invalid detector efficiency or noise")


@dataclass(frozen=True)
class Run:
    """输运方法、随机种子、采样量和 Geant4 数值步长。

``n_simulated`` 是用于估计落点分布的 Monte Carlo 历史数；
``exposure_electrons`` 是拟合的实验曝光电子数，两者不能混为一谈。
``droplet_material='vacuum'`` 保留电场却移去水的散射/能损，供对照使用。
"""

    engine: str = "ray"  # ray, ideal_occluder, geant4
    n_simulated: int = 100000
    exposure_electrons: int = 100000
    seed: int = 20260923
    raster_width_mm: float = 12.0
    transport_step_um: float = 10.0
    geant4_executable: str | None = None
    droplet_material: str = "water"  # water or vacuum
    geant4_world_step_mm: float = 5.0
    geant4_far_step_mm: float = 0.25
    geant4_near_step_um: float = 10.0
    geant4_core_step_um: float = 0.5
    geant4_cut_um: float = 1.0
    geant4_layer_step_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.engine not in {"ray", "ideal_occluder", "geant4"}:
            raise ValueError("engine must be ray, ideal_occluder or geant4")
        if self.droplet_material not in {"water", "vacuum"}:
            raise ValueError("droplet_material must be water or vacuum")
        if min(self.n_simulated, self.exposure_electrons, self.raster_width_mm,
               self.transport_step_um, self.geant4_world_step_mm,
               self.geant4_far_step_mm, self.geant4_near_step_um,
               self.geant4_core_step_um, self.geant4_cut_um,
               self.geant4_layer_step_scale) <= 0:
            raise ValueError("Run counts and widths must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    """一次模拟所需的完整配置，所有子配置共享同一组默认值。"""

    geometry: Geometry = field(default_factory=Geometry)
    charge: Charge = field(default_factory=Charge)
    source: Source = field(default_factory=Source)
    detector: Detector = field(default_factory=Detector)
    run: Run = field(default_factory=Run)

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 JSON 的普通字典，供结果溯源和缓存比较。"""
        return asdict(self)

    def with_updates(self, **sections: dict[str, Any]) -> "SimulationConfig":
        """只替换指定分组的指定字段；原配置保持不变。"""
        updated = self
        for section, changes in sections.items():
            updated = replace(updated, **{section: replace(getattr(updated, section), **changes)})
        return updated


def load_config(path: str | Path | None = None) -> SimulationConfig:
    """读取 YAML，并通过各 dataclass 构造函数统一验证参数。"""
    raw = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = {"geometry": Geometry, "charge": Charge, "source": Source,
             "detector": Detector, "run": Run}
    unexpected = set(raw) - set(known)
    if unexpected:
        raise ValueError(f"Unknown configuration sections: {sorted(unexpected)}")
    return SimulationConfig(**{key: cls(**raw.get(key, {})) for key, cls in known.items()})
