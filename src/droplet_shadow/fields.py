"""液滴电势/电场模型；所有位置、电荷和输出都使用 SI 单位。

``RadialShells``/``UniformVolume`` 给出球对称解析场；
``poisson_boltzmann`` 自洽计算球内移动离子。界面介电层存在时，
所有固定电荷分量共用同一分层介电分布，保证叠加后的边界条件一致。
所有模型实现同一个约定：电势在无穷远为零，电场 ``E=-∇φ``，
``points`` 的最后一维是 (x,y,z)，可批量传入任意前导维度。
这里的连续介质平均场不等同于单个水分子 O–H 键上的瞬时局域场。
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import math
from typing import Protocol

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicHermiteSpline
from .config import SimulationConfig
from .constants import COULOMB_K, E_CHARGE, EPS0, K_B


def _points(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """将任意形状 ``(...,3)`` 展成 ``(N,3)``，并保存原来的批量形状。"""
    a = np.asarray(points, dtype=float)
    if a.ndim == 0 or a.shape[-1] != 3:
        raise ValueError("Points must have a final axis of length 3")
    return a.reshape(-1, 3), a.shape[:-1]


def rayleigh_charge_e(radius_m: float, surface_tension_n_m: float) -> float:
    """Rayleigh 裂变极限电荷，以元电荷数表示；只作尺度比较。

``Q_R=8π√(ε₀γR³)`` 假定理想液滴的静电与表面张力竞争，
不是模型自动施加的电荷上限。
"""
    return 8 * math.pi * math.sqrt(EPS0 * surface_tension_n_m * radius_m**3) / E_CHARGE


class Field(Protocol):
    """电场接口：输入 m，返回电势 V 与电场 V/m。"""

    def potential(self, points: np.ndarray) -> np.ndarray:
        """返回每个输入点的电势 φ（V）。"""
        ...

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """返回每个输入点的三分量电场 E（V/m）。"""
        ...


def _regions(radius: float, eps_in: float, eps_out: float,
             layer_inner: float | None, eps_layer: float | None) -> list[tuple[float, float, float]]:
    """返回 (内半径, 外半径, 相对介电常数)，供所有固定电荷分量共用。

无界面介电层时整个球内使用 eps_in；存在层时，内核使用 eps_in，
最外一层使用 eps_layer。介电边界本身不自动增加自由电荷。
"""
    if not all(np.isfinite(v) and v > 0 for v in (radius, eps_in, eps_out)):
        raise ValueError("Radius and relative permittivities must be finite and positive")
    if layer_inner is None and eps_layer is None:
        return [(0.0, radius, eps_in)]
    if (layer_inner is None or eps_layer is None or
            not np.isfinite(layer_inner) or not 0 < layer_inner < radius or
            not np.isfinite(eps_layer) or eps_layer <= 0):
        raise ValueError("Dielectric layer requires 0 < inner radius < R and positive permittivity")
    return [(0.0, layer_inner, eps_in), (layer_inner, radius, eps_layer)]


def _permittivity(r: np.ndarray, radius: float, eps_out: float,
                  regions: list[tuple[float, float, float]]) -> np.ndarray:
    """边界点统一取外侧极限；r=R 使用外部介电常数。"""
    eps = np.full_like(r, eps_out)
    for low, high, value in regions:
        eps[(r >= low) & (r < high)] = value
    return eps


@dataclass(frozen=True)
class RadialShells:
    """同心带电薄壳，可表示表面电荷、中性双层或等效偶极层。

``shell_charges_c`` 是每个壳的**总电荷** C；不同半径的面密度
``σ=Q/(4πr²)`` 并不相同。内核用 ``epsilon_inside``，可选界面层用
``epsilon_layer``，球外用 ``epsilon_outside``（均为相对介电常数）。球面场存在跳变，调用方不应
把插值跨越壳面当作连续场。
"""

    radius_m: float
    shell_radii_m: tuple[float, ...]
    shell_charges_c: tuple[float, ...]
    epsilon_inside: float = 78.0
    epsilon_outside: float = 1.0
    layer_inner_m: float | None = None
    epsilon_layer: float | None = None

    def __post_init__(self) -> None:
        """检查薄壳从内到外排序，且全部落在液滴内部或表面。"""
        _regions(self.radius_m, self.epsilon_inside, self.epsilon_outside,
                 self.layer_inner_m, self.epsilon_layer)
        if not self.shell_radii_m or len(self.shell_radii_m) != len(self.shell_charges_c):
            raise ValueError("Shell radii and charges must be nonempty and equally sized")
        if any(not np.isfinite(r) or r <= 0 or r > self.radius_m for r in self.shell_radii_m):
            raise ValueError("All shells must lie inside or on the droplet")
        if not np.all(np.isfinite(self.shell_charges_c)):
            raise ValueError("Shell charges must be finite")
        if list(self.shell_radii_m) != sorted(self.shell_radii_m):
            raise ValueError("Shell radii must be sorted")

    @property
    def total_charge_c(self) -> float:
        """外部 Coulomb 场只由所有薄壳电荷之和决定。"""
        return math.fsum(self.shell_charges_c)

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """径向电势与场；先从无穷远积分势，再由高斯定律求场。"""
        q = np.zeros_like(r)
        regions = _regions(self.radius_m, self.epsilon_inside, self.epsilon_outside,
                           self.layer_inner_m, self.epsilon_layer)
        # 先设 φ(R)，再对每个“包围电荷和介电常数均恒定”的区间积分。
        # 电荷壳面、介电界面都必须成为积分分段点。
        phi = np.full_like(r, COULOMB_K * self.total_charge_c / (self.epsilon_outside * self.radius_m))
        edges = sorted(set((*self.shell_radii_m, self.radius_m,
                            *([self.layer_inner_m] if self.layer_inner_m is not None else []))))
        for inner, outer in zip(edges[:-1], edges[1:]):
            enclosed = math.fsum(qi for ri, qi in zip(self.shell_radii_m, self.shell_charges_c)
                                  if ri <= inner)
            eps = _permittivity(np.array([inner]), self.radius_m, self.epsilon_outside, regions)[0]
            r_clip = np.clip(r, inner, outer)
            # 等价于 1/r_clip - 1/outer；相减距离可避免纳米薄层的相消误差。
            phi += COULOMB_K * enclosed / eps * (outer - r_clip) / (r_clip * outer)
        outside = r >= self.radius_m
        phi[outside] = COULOMB_K * self.total_charge_c / (self.epsilon_outside * r[outside])
        for radius, charge in zip(self.shell_radii_m, self.shell_charges_c):
            # 半径 r 内已包围的总电荷；r=0 处用 where 避免除零。
            q += charge * (r >= radius)
        eps = _permittivity(r, self.radius_m, self.epsilon_outside, regions)
        e_rad = np.divide(COULOMB_K * q, eps * r**2, out=np.zeros_like(r), where=r > 0)
        return phi, e_rad

    def potential(self, points: np.ndarray) -> np.ndarray:
        """求任意三维点的电势（V），保持输入的批量维度。"""
        p, shape = _points(points)
        return self._radial(np.linalg.norm(p, axis=1))[0].reshape(shape)

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """将径向场 ``E_r`` 沿 ``r̂`` 投影回笛卡尔三分量。"""
        p, shape = _points(points)
        r = np.linalg.norm(p, axis=1)
        er = self._radial(r)[1]
        out = p * np.divide(er, r, out=np.zeros_like(r), where=r > 0)[:, None]
        return out.reshape((*shape, 3))


@dataclass(frozen=True)
class UniformVolume:
    """规定的均匀体电荷分布，用来与同 Q 的表面壳比较。

球外场必与同总电荷的表面壳相同；球内包围电荷按 r³ 增长，
所以在介电常数恒定的区域 ``E_r∝r``。这只是规定分布的物理对照，
不宣称是水滴的离子平衡态。
"""

    radius_m: float
    charge_c: float
    epsilon_inside: float = 78.0
    epsilon_outside: float = 1.0
    layer_inner_m: float | None = None
    epsilon_layer: float | None = None

    def __post_init__(self) -> None:
        _regions(self.radius_m, self.epsilon_inside, self.epsilon_outside,
                 self.layer_inner_m, self.epsilon_layer)
        if not np.isfinite(self.charge_c):
            raise ValueError("Volume charge must be finite")

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """解析求解介电球内外的电势和电场，并在 r=R 连续接势。"""
        R = self.radius_m
        inside = r < R
        regions = _regions(R, self.epsilon_inside, self.epsilon_outside,
                           self.layer_inner_m, self.epsilon_layer)
        phi = COULOMB_K * self.charge_c / (self.epsilon_outside * np.maximum(r, R))
        for low, high, eps in regions:
            clipped = np.clip(r, low, high)
            phi += COULOMB_K * self.charge_c / R**3 * (high - clipped) * (high + clipped) / (2 * eps)
        er = COULOMB_K * self.charge_c / (
            self.epsilon_outside * np.maximum(r, R) ** 2
        )
        eps = _permittivity(r, R, self.epsilon_outside, regions)
        er[inside] = COULOMB_K * self.charge_c * r[inside] / (eps[inside] * R**3)
        return phi, er

    def potential(self, points: np.ndarray) -> np.ndarray:
        """对任意批量三维点返回体电荷模型的电势（V）。"""
        p, shape = _points(points)
        return self._radial(np.linalg.norm(p, axis=1))[0].reshape(shape)

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """返回体电荷模型的笛卡尔电场（V/m）。"""
        p, shape = _points(points)
        r = np.linalg.norm(p, axis=1)
        er = self._radial(r)[1]
        return (p * np.divide(er, r, out=np.zeros_like(r), where=r > 0)[:, None]).reshape((*shape, 3))


@dataclass(frozen=True)
class TabulatedRadial:
    """由 PB 数值解产生的径向查表场；球外仍接解析净电荷场。

表内 ``radii_m`` 在界面附近自适应加密。用电势及其斜率 -E 构造
Hermite 插值；查询电场时对同一插值求导，保证 E=-dφ/dr。
"""

    radius_m: float
    radii_m: np.ndarray
    potential_v: np.ndarray
    field_v_m: np.ndarray
    charge_c: float
    epsilon_outside: float = 1.0
    self_consistent: bool = False
    ion_counts: tuple[float, ...] = ()
    _spline: CubicHermiteSpline = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """表必须覆盖精确的 [0,R]，最后一行场值保存 R 的内侧极限。"""
        r = np.asarray(self.radii_m, dtype=float)
        phi = np.asarray(self.potential_v, dtype=float)
        er = np.asarray(self.field_v_m, dtype=float)
        if (r.ndim != 1 or len(r) < 2 or phi.shape != r.shape or er.shape != r.shape or
                not all(np.all(np.isfinite(a)) for a in (r, phi, er)) or
                r[0] != 0 or r[-1] != self.radius_m or np.any(np.diff(r) <= 0)):
            raise ValueError("Radial table must be finite, strictly increasing, and cover [0,R]")
        object.__setattr__(self, "_spline", CubicHermiteSpline(r, phi, -er, extrapolate=False))

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """液滴内插值 PB 数值表，液滴外使用总净电荷解析场。"""
        inside = r < self.radius_m
        phi = np.empty_like(r)
        er = np.empty_like(r)
        phi[inside] = self._spline(r[inside])
        er[inside] = -self._spline(r[inside], 1)
        phi[~inside] = COULOMB_K * self.charge_c / (self.epsilon_outside * r[~inside])
        er[~inside] = COULOMB_K * self.charge_c / (self.epsilon_outside * r[~inside] ** 2)
        return phi, er

    def potential(self, points: np.ndarray) -> np.ndarray:
        """返回与输入点批量形状一致的电势数组。"""
        p, shape = _points(points)
        return self._radial(np.linalg.norm(p, axis=1))[0].reshape(shape)

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """把插值的径向场变为笛卡尔电场数组。"""
        p, shape = _points(points)
        r = np.linalg.norm(p, axis=1)
        er = self._radial(r)[1]
        return (p * np.divide(er, r, out=np.zeros_like(r), where=r > 0)[:, None]).reshape((*shape, 3))


def poisson_boltzmann(config: SimulationConfig, nodes: int = 320,
                      tol: float = 1e-6) -> TabulatedRadial:
    """求球坐标、固定离子总数的 Poisson–Boltzmann 平均场。

设 ``x=r/R``、``u=e(φ-φ(R))/(k_B T)``；对每种价态 z=±1 的离子，
密度满足 ``n_z(x)∝exp(-z u(x))``。比例常数不是任意指定的体浓度，
而是作为 ``solve_bvp`` 的未知参数，与积分态变量一起求，使每种
离子的球内积分恰好等于给定粒子数。固定表面电荷不属于移动离子。
相对表面势求解可避免带电滴数百伏公共偏置造成数值相消；输出时再
加回 φ(R)，仍以无穷远为零。返回连续介质平均场，不能解析分子层结构。
"""
    if not isinstance(nodes, int) or nodes < 16 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("PB requires at least 16 initial nodes and positive tolerance")
    g, ch = config.geometry, config.charge
    if ch.double_layer_q_e or ch.dipole_potential_v:
        raise ValueError("PB plus interface layers requires a coupled ion re-solve")
    R = g.radius_m
    species = [(1, ch.ion_positive_count), (-1, ch.ion_negative_count)]
    species = [(z, n) for z, n in species if n > 0]
    expected_qe = ch.surface_fixed_e + ch.ion_positive_count - ch.ion_negative_count
    # 这个守恒关系防止用户把表面电荷、反离子和“净 Q”重复计算。
    if abs(expected_qe - ch.q_e) > 1e-7 * max(1.0, abs(ch.q_e)):
        raise ValueError("PB q_e must equal surface_fixed_e + positive ions - negative ions")
    if not species:
        return TabulatedRadial(R, np.array([0, R]),
                               np.full(2, COULOMB_K * ch.q_e * E_CHARGE / R),
                               np.zeros(2), ch.q_e * E_CHARGE, self_consistent=True)
    nref = max(n for _, n in species)
    # u_edge 是输出时加回的绝对势偏置；求解中的相对表面势边界为零。
    u_edge = E_CHARGE * COULOMB_K * ch.q_e * E_CHARGE / (K_B * g.temperature_k * R)
    beta = 3 * E_CHARGE**2 * nref / (4 * math.pi * EPS0 * g.epsilon_water * K_B * g.temperature_k * R)
    x = np.unique(np.r_[np.linspace(1e-6, 0.95, nodes // 2),
                          1 - np.geomspace(1e-7, 0.05, nodes // 2)[::-1], 1.0])
    # 必须包含精确 x=1；否则边界条件被施加在球面内侧，表末尾会被常值外推。
    # x≈1 的点更密，以解析受表面固定电荷吸引的反离子层。
    charges = np.array([z for z, _ in species])
    targets = np.array([n / nref for _, n in species])

    def fun(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # y[0]=相对表面的 u，y[1]=x² du/dx；y[2:] 为逐种离子的归一化累积数。
        # p 是各离子的对数归一化系数，避免直接求正数幅值。
        dens = np.exp(np.clip(p[:, None] - charges[:, None] * y[0], -100, 100))
        return np.vstack((y[1] / t**2,
                          -beta * t**2 * (charges[:, None] * dens).sum(axis=0),
                          3 * t[None, :] ** 2 * dens))

    def bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
        # x_min=1e-6 近似中心；省略的有限密度内核体积比例为 1e-18。
        # 在精确 x=1 令相对表面势为零，并固定每类离子的累计总数。
        return np.r_[ya[1], ya[2:], yb[0], yb[2:] - targets]

    y0 = np.zeros((2 + len(species), x.size))
    # 零相对势 + 均匀离子分布作为初猜；p 是相对势规范下的归一化系数。
    y0[2:] = targets[:, None] * x[None, :] ** 3
    p0 = np.log(targets)
    sol = solve_bvp(fun, bc, x, y0, p=p0, tol=tol,
                    max_nodes=20000, verbose=0)
    if not sol.success:
        raise RuntimeError(f"Poisson-Boltzmann solver failed: {sol.message}")
    # clip 只帮助迭代避免溢出；收敛解若仍触及截断，就不再是原 PB 方程。
    exponent = sol.p[:, None] - charges[:, None] * sol.y[0]
    if np.any(np.abs(exponent) >= 100):
        raise RuntimeError("PB solution reaches the exponential clipping limit")
    radii = np.r_[0.0, sol.x * R]
    u = np.r_[sol.y[0, 0], sol.y[0]] + u_edge
    # y[1]/x²=du/dx；E=-dφ/dr，中心因球对称严格为 0。
    er = -K_B * g.temperature_k / (E_CHARGE * R) * np.r_[0.0, sol.y[1] / sol.x**2]
    return TabulatedRadial(R, radii, u * K_B * g.temperature_k / E_CHARGE,
                           er, ch.q_e * E_CHARGE, self_consistent=True,
                           ion_counts=tuple(sol.y[2:, -1] * nref))


class SumField:
    """把固定场分量线性叠加；不得绕开 PB 的耦合约束。"""

    def __init__(self, *fields: Field):
        """展开固定场组合，并拒绝未经离子重求解的 PB 叠加。

解析场必须使用相同的径向介电分布，才能叠加为同一个 Poisson 问题。
"""
        pieces = tuple(piece for f in fields for piece in (f.fields if isinstance(f, SumField) else (f,)))
        if len(pieces) > 1 and any(isinstance(f, TabulatedRadial) and f.self_consistent for f in pieces):
            raise ValueError("PB superposition requires a coupled ion re-solve")
        media = {(f.radius_m, f.epsilon_inside, f.epsilon_outside, f.layer_inner_m, f.epsilon_layer)
                 for f in pieces if isinstance(f, (RadialShells, UniformVolume))}
        if len(media) > 1:
            raise ValueError("Fixed fields must share the same radial dielectric profile")
        self.fields = pieces

    def potential(self, points: np.ndarray) -> np.ndarray:
        """叠加各分量电势；即使没有分量也返回零数组。"""
        return sum((f.potential(points) for f in self.fields), start=np.zeros(np.asarray(points).shape[:-1]))

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """叠加各分量电场，输出保持与点数组相同的三维形状。"""
        return sum((f.electric_field(points) for f in self.fields), start=np.zeros_like(np.asarray(points, dtype=float)))


def build_field(config: SimulationConfig) -> Field:
    """把配置转换成最终场对象，统一组合净电荷和可叠加界面层。"""
    g, c = config.geometry, config.charge
    R = g.radius_m
    # 激活偶极层时，同时指定真实的径向介电层；体电荷、双层及偶极层
    # 都在这一个介电环境中求解。不能只给偶极分量使用另一套介电常数。
    medium = {}
    if c.dipole_potential_v:
        inner = R - c.dipole_thickness_nm * 1e-9
        if not 0 < inner < R:
            raise ValueError("Dipole layer thickness must be resolvable and smaller than R")
        medium = {"layer_inner_m": inner, "epsilon_layer": c.dipole_epsilon}
    if c.model == "surface":
        base: Field = RadialShells(R, (R,), (c.q_e * E_CHARGE,), g.epsilon_water, **medium)
    elif c.model == "volume":
        base = UniformVolume(R, c.q_e * E_CHARGE, g.epsilon_water, **medium)
    elif c.model == "neutral_double_layer":
        # 半径不同的 +Q/-Q 薄壳在球外严格抵消；额外净 Q 单独添加。
        inner = R - c.double_layer_thickness_nm * 1e-9
        if not 0 < inner < R:
            raise ValueError("Double layer is thicker than the droplet radius")
        base = RadialShells(R, (inner, R),
                            (c.double_layer_q_e * E_CHARGE, -c.double_layer_q_e * E_CHARGE),
                            g.epsilon_water, **medium)
        if c.q_e:
            base = SumField(base, RadialShells(R, (R,), (c.q_e * E_CHARGE,), g.epsilon_water, **medium))
    elif c.model == "poisson_boltzmann":
        base = poisson_boltzmann(config)
    else:
        raise ValueError(f"Unknown spherical charge model: {c.model}")
    fields: list[Field] = [base]
    if c.model in {"surface", "volume"} and c.double_layer_q_e:
        inner = R - c.double_layer_thickness_nm * 1e-9
        if not 0 < inner < R:
            raise ValueError("Double layer is thicker than the droplet radius")
        fields.append(RadialShells(R, (inner, R),
                                   (c.double_layer_q_e * E_CHARGE,
                                    -c.double_layer_q_e * E_CHARGE),
                                   g.epsilon_water, **medium))
    if c.dipole_potential_v:
        # 用两壳之间的势差 Δφ 反算等效总电荷；取向层的净电荷仍为零。
        inner = R - c.dipole_thickness_nm * 1e-9
        if inner <= 0:
            raise ValueError("Dipole layer is thicker than the droplet radius")
        # 1/inner - 1/R = (R-inner)/(inner*R)，后者在极薄层中更稳定。
        q = c.dipole_potential_v * c.dipole_epsilon * inner * R / (COULOMB_K * (R - inner))
        fields.append(RadialShells(R, (inner, R), (q, -q), g.epsilon_water, **medium))
    return fields[0] if len(fields) == 1 else SumField(*fields)
