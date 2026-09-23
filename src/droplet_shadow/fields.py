"""液滴电势/电场模型；所有位置、电荷和输出都使用 SI 单位。

模块按复杂度分三层：``RadialShells``/``UniformVolume`` 是可解析的
球对称对照；``poisson_boltzmann`` 计算球内可移动离子；
``SurfaceMultipoles`` 描述暂时冻结的非对称表面电荷。
所有模型实现同一个约定：电势在无穷远为零，电场 ``E=-∇φ``，
``points`` 的最后一维是 (x,y,z)，可批量传入任意前导维度。
这里的连续介质平均场不等同于单个水分子 O–H 键上的瞬时局域场。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
from scipy.integrate import solve_bvp
try:
    from scipy.special import sph_harm_y as _sph_harm_y

    def spherical_harmonic(m: int, ell: int, azimuth: np.ndarray, polar: np.ndarray) -> np.ndarray:
        """统一 SciPy 新版球谐函数的角度顺序：方位角 φ、极角 θ。"""
        return _sph_harm_y(ell, m, polar, azimuth)
except ImportError:
    from scipy.special import sph_harm as _sph_harm

    def spherical_harmonic(m: int, ell: int, azimuth: np.ndarray, polar: np.ndarray) -> np.ndarray:
        """兼容旧版 ``sph_harm(m,l,φ,θ)`` 接口。"""
        return _sph_harm(m, ell, azimuth, polar)

from .config import SimulationConfig
from .constants import COULOMB_K, E_CHARGE, EPS0, K_B


def _points(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """将任意形状 ``(...,3)`` 展成 ``(N,3)``，并保存原来的批量形状。"""
    a = np.asarray(points, dtype=float)
    if a.shape[-1] != 3:
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


@dataclass(frozen=True)
class RadialShells:
    """同心带电薄壳，可表示表面电荷、中性双层或等效偶极层。

``shell_charges_c`` 是每个壳的**总电荷** C；不同半径的面密度
``σ=Q/(4πr²)`` 并不相同。壳间用 ``ε_inside``，球外用
``ε_outside``（均为相对介电常数）。球面场存在跳变，调用方不应
把插值跨越壳面当作连续场。
"""

    radius_m: float
    shell_radii_m: tuple[float, ...]
    shell_charges_c: tuple[float, ...]
    epsilon_inside: float = 78.0
    epsilon_outside: float = 1.0

    def __post_init__(self) -> None:
        """检查薄壳从内到外排序，且全部落在液滴内部或表面。"""
        if not self.shell_radii_m or len(self.shell_radii_m) != len(self.shell_charges_c):
            raise ValueError("Shell radii and charges must be nonempty and equally sized")
        if any(r <= 0 or r > self.radius_m for r in self.shell_radii_m):
            raise ValueError("All shells must lie inside or on the droplet")
        if list(self.shell_radii_m) != sorted(self.shell_radii_m):
            raise ValueError("Shell radii must be sorted")

    @property
    def total_charge_c(self) -> float:
        """外部 Coulomb 场只由所有薄壳电荷之和决定。"""
        return sum(self.shell_charges_c)

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """径向电势与场；先从无穷远积分势，再由高斯定律求场。"""
        q = np.zeros_like(r)
        # 球外 φ=kQ/(ε_out r)；在 R 处先设势，再对每个壳间区间向内积分
        # E= k Q_enc/(ε_in r²)。对于 +Q/-Q 中性双层，外势/外场严格为零。
        phi = np.full_like(r, COULOMB_K * self.total_charge_c / (self.epsilon_outside * self.radius_m))
        for i in range(len(self.shell_radii_m) - 1, -1, -1):
            outer = self.radius_m if i == len(self.shell_radii_m) - 1 else self.shell_radii_m[i + 1]
            inner = self.shell_radii_m[i]
            enclosed = sum(self.shell_charges_c[: i + 1])
            # clip 让当前区间仅对位于其内侧的点贡献正确的势差。
            r_clip = np.clip(r, inner, outer)
            phi += COULOMB_K * enclosed / self.epsilon_inside * (1 / r_clip - 1 / outer)
        outside = r >= self.radius_m
        phi[outside] = COULOMB_K * self.total_charge_c / (self.epsilon_outside * r[outside])
        for radius, charge in zip(self.shell_radii_m, self.shell_charges_c):
            # 半径 r 内已包围的总电荷；r=0 处用 where 避免除零。
            q += charge * (r >= radius)
        eps = np.where(outside, self.epsilon_outside, self.epsilon_inside)
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
所以 ``E_r∝r``。这只是物理对照，不宣称是水滴的离子平衡态。
"""

    radius_m: float
    charge_c: float
    epsilon_inside: float = 78.0
    epsilon_outside: float = 1.0

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """解析求解介电球内外的电势和电场，并在 r=R 连续接势。"""
        R = self.radius_m
        inside = r < R
        phi = COULOMB_K * self.charge_c / (self.epsilon_outside * np.maximum(r, R))
        phi[inside] = COULOMB_K * self.charge_c * (
            1 / (self.epsilon_outside * R) + (R**2 - r[inside] ** 2) / (2 * self.epsilon_inside * R**3)
        )
        er = COULOMB_K * self.charge_c / (
            self.epsilon_outside * np.maximum(r, R) ** 2
        )
        er[inside] = COULOMB_K * self.charge_c * r[inside] / (self.epsilon_inside * R**3)
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

表内 ``radii_m`` 可在界面附近自适应加密；插值用于轨迹采样，
不是在整个毫米传播区做纳米三维网格。
"""

    radius_m: float
    radii_m: np.ndarray
    potential_v: np.ndarray
    field_v_m: np.ndarray
    charge_c: float
    epsilon_outside: float = 1.0

    def _radial(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """液滴内插值 PB 数值表，液滴外使用总净电荷解析场。"""
        inside = r <= self.radius_m
        phi = np.empty_like(r)
        er = np.empty_like(r)
        phi[inside] = np.interp(r[inside], self.radii_m, self.potential_v)
        er[inside] = np.interp(r[inside], self.radii_m, self.field_v_m)
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


def poisson_boltzmann(config: SimulationConfig, nodes: int = 320) -> TabulatedRadial:
    """求球坐标、固定离子总数的 Poisson–Boltzmann 平均场。

设 ``x=r/R``、``u=eφ/(k_B T)``；对每种价态 z=±1 的离子，
密度满足 ``n_z(x)∝exp(-z u(x))``。比例常数不是任意指定的体浓度，
而是作为 ``solve_bvp`` 的未知参数，与积分态变量一起求，使每种
离子的球内积分恰好等于给定粒子数。固定表面电荷不属于移动离子。
返回的表只描述连续介质平均场，不能解析分子层结构。
"""
    g, ch = config.geometry, config.charge
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
                               np.zeros(2), ch.q_e * E_CHARGE)
    nref = max(n for _, n in species)
    # u_edge 由球外总净电荷给定；beta 是无量纲 PB 耦合强度。
    u_edge = E_CHARGE * COULOMB_K * ch.q_e * E_CHARGE / (K_B * g.temperature_k * R)
    beta = 3 * E_CHARGE**2 * nref / (4 * math.pi * EPS0 * g.epsilon_water * K_B * g.temperature_k * R)
    x = np.unique(np.r_[np.linspace(1e-6, 0.95, nodes // 2),
                          1 - np.geomspace(1e-7, 0.05, nodes // 2)[::-1]])
    # x≈1 的点更密，以解析受表面固定电荷吸引的反离子层。
    charges = np.array([z for z, _ in species])
    targets = np.array([n / nref for _, n in species])

    def fun(t: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # y[0]=u，y[1]=x² du/dx；y[2:] 为逐种离子的归一化累积数。
        # p 是各离子的对数归一化系数，避免直接求正数幅值。
        dens = np.exp(np.clip(p[:, None] - charges[:, None] * y[0], -100, 100))
        return np.vstack((y[1] / t**2,
                          -beta * t**2 * (charges[:, None] * dens).sum(axis=0),
                          3 * t[None, :] ** 2 * dens))

    def bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
        # 中心球对称：x²u'=0，累积数=0；边界电势接外场，累积数=目标值。
        return np.r_[ya[1], ya[2:], yb[0] - u_edge, yb[2:] - targets]

    y0 = np.zeros((2 + len(species), x.size))
    y0[0] = u_edge
    y0[2:] = targets[:, None] * x[None, :] ** 3
    p0 = np.log(targets) + charges * u_edge
    sol = solve_bvp(fun, bc, x, y0, p=p0, tol=2e-4,
                    max_nodes=20000, verbose=0)
    if not sol.success:
        raise RuntimeError(f"Poisson-Boltzmann solver failed: {sol.message}")
    radii = np.r_[0.0, sol.x * R]
    u = np.r_[sol.y[0, 0], sol.y[0]]
    # y[1]/x²=du/dx；E=-dφ/dr，中心因球对称严格为 0。
    er = -K_B * g.temperature_k / (E_CHARGE * R) * np.r_[0.0, sol.y[1] / sol.x**2]
    return TabulatedRadial(R, radii, u * K_B * g.temperature_k / E_CHARGE,
                           er, ch.q_e * E_CHARGE)


class SumField:
    """把固定场分量线性叠加；不得绕开 PB 的耦合约束。"""

    def __init__(self, *fields: Field):
        """保存将被线性相加的、已经通过配置约束检查的场。"""
        self.fields = tuple(fields)

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
    if c.model == "surface":
        base: Field = RadialShells(R, (R,), (c.q_e * E_CHARGE,), g.epsilon_water)
    elif c.model == "volume":
        base = UniformVolume(R, c.q_e * E_CHARGE, g.epsilon_water)
    elif c.model == "neutral_double_layer":
        # 半径不同的 +Q/-Q 薄壳在球外严格抵消；额外净 Q 单独添加。
        inner = R - c.double_layer_thickness_nm * 1e-9
        if inner <= 0:
            raise ValueError("Double layer is thicker than the droplet radius")
        base = RadialShells(R, (inner, R),
                            (c.double_layer_q_e * E_CHARGE, -c.double_layer_q_e * E_CHARGE),
                            g.epsilon_water)
        if c.q_e:
            base = SumField(base, RadialShells(R, (R,), (c.q_e * E_CHARGE,), g.epsilon_water))
    else:
        base = poisson_boltzmann(config)
    fields: list[Field] = [base]
    if c.model in {"surface", "volume"} and c.double_layer_q_e:
        inner = R - c.double_layer_thickness_nm * 1e-9
        if inner <= 0:
            raise ValueError("Double layer is thicker than the droplet radius")
        fields.append(RadialShells(R, (inner, R),
                                   (c.double_layer_q_e * E_CHARGE,
                                    -c.double_layer_q_e * E_CHARGE),
                                   g.epsilon_water))
    if c.dipole_potential_v:
        # 用两壳之间的势差 Δφ 反算等效总电荷；取向层的净电荷仍为零。
        inner = R - c.dipole_thickness_nm * 1e-9
        if inner <= 0:
            raise ValueError("Dipole layer is thicker than the droplet radius")
        q = c.dipole_potential_v * c.dipole_epsilon / (COULOMB_K * (1 / inner - 1 / R))
        fields.append(RadialShells(R, (inner, R), (q, -q), c.dipole_epsilon))
    if c.dipole_fraction or c.quadrupole_fraction or c.patch_fraction:
        fields.append(SurfaceMultipoles(config))
    return fields[0] if len(fields) == 1 else SumField(*fields)


class SurfaceMultipoles:
    """零单极矩的冻结表面电荷起伏：偶极、四极和局部斑块。

先在球面数值积分 ``σ_lm=∫σ(θ,φ)Y*_lm dΩ``，再用球面势连续及
位移矢量跳变条件求每个球谐的内外势系数。``l=0`` 刻意留给
``q_e`` 的球对称基场，因此非对称项不改变液滴总净电荷。
"""

    def __init__(self, config: SimulationConfig):
        """球面求积并求出至 ``multipole_order`` 的非单极势系数。"""
        g, c = config.geometry, config.charge
        self.radius_m = g.radius_m
        self.epsilon_inside = g.epsilon_water * EPS0
        self.epsilon_outside = EPS0
        order = c.multipole_order
        u, w = np.polynomial.legendre.leggauss(max(2 * order + 8, 32))
        # Gauss–Legendre 积分变量 u=cosθ，方位角 φ 用等间距采样。
        phi = np.linspace(0, 2 * np.pi, max(4 * order + 16, 64), endpoint=False)
        uu, pp = np.meshgrid(u, phi, indexing="ij")
        nx = np.sqrt(1 - uu**2) * np.cos(pp)
        sigma0 = c.q_e * E_CHARGE / (4 * np.pi * self.radius_m**2)
        patch_center = np.cos(np.arccos(np.clip(nx, -1, 1)))
        patch = np.exp((patch_center - 1) / (2 * np.sin(np.deg2rad(c.patch_width_deg) / 2) ** 2))
        patch -= np.average(patch, weights=np.broadcast_to(w[:, None], patch.shape))
        # 从斑块减去球面平均，保证它没有额外单极矩。
        patch_base = (abs(c.q_e) if c.q_e else 1e6) * E_CHARGE / (4 * np.pi * self.radius_m**2)
        sigma = sigma0 * (c.dipole_fraction * nx + c.quadrupole_fraction * (3 * nx**2 - 1) / 2)
        sigma += c.patch_fraction * patch_base * patch
        self.coefficients: dict[tuple[int, int], complex] = {}
        weights = w[:, None] * (2 * np.pi / phi.size)
        theta = np.arccos(uu)
        for ell in range(1, order + 1):
            for m in range(-ell, ell + 1):
                value = np.sum(sigma * np.conj(spherical_harmonic(m, ell, pp, theta)) * weights)
                if abs(value) > 1e-16 * max(abs(sigma0), abs(patch_base)):
                    # 分母来自 ε_out(l+1)+ε_in l；满足球面介电边界条件。
                    self.coefficients[ell, m] = value * self.radius_m ** (ell + 2) / (
                        self.epsilon_outside * (ell + 1) + self.epsilon_inside * ell)

    def potential(self, points: np.ndarray) -> np.ndarray:
        """球内势随 r^l、球外势随 r^(-l-1) 变化。"""
        xyz, shape = _points(points)
        r = np.linalg.norm(xyz, axis=1)
        theta = np.arccos(np.divide(xyz[:, 2], r, out=np.ones_like(r), where=r > 0))
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        result = np.zeros(len(r), dtype=complex)
        inside = r < self.radius_m
        for (ell, m), a in self.coefficients.items():
            angular = spherical_harmonic(m, ell, azimuth, theta)
            radial = np.empty_like(r, dtype=complex)
            radial[inside] = a * (r[inside] / self.radius_m) ** ell / self.radius_m ** (ell + 1)
            radial[~inside] = a / r[~inside] ** (ell + 1)
            result += radial * angular
        return result.real.reshape(shape)

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """用电势有限差分求 ``-∇φ``，跨界面时改用单侧差分。"""
        xyz, shape = _points(points)
        h = self.radius_m * 2e-5
        central_inside = np.linalg.norm(xyz, axis=1) < self.radius_m
        phi0 = self.potential(xyz)
        grads = []
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = h
            plus = xyz + delta
            minus = xyz - delta
            phi_plus = self.potential(plus)
            phi_minus = self.potential(minus)
            derivative = (phi_plus - phi_minus) / (2 * h)
            plus_crosses = (np.linalg.norm(plus, axis=1) < self.radius_m) != central_inside
            minus_crosses = (np.linalg.norm(minus, axis=1) < self.radius_m) != central_inside
            derivative = np.where(plus_crosses & ~minus_crosses,
                                  (phi0 - phi_minus) / h, derivative)
            derivative = np.where(minus_crosses & ~plus_crosses,
                                  (phi_plus - phi0) / h, derivative)
            grads.append(derivative)
        return -np.stack(grads, axis=-1).reshape((*shape, 3))


def multipole_convergence_error(config: SimulationConfig) -> float | None:
    """将球谐截断阶数提高 4 后，比较球外测试点的场 RMS 相对变化。

这个数越大，说明局部斑块仍需更多高阶球谐；它是数值截断诊断，
不是从单张 shadow 反演非对称三维分布的置信度。
"""
    charge = config.charge
    if not (charge.dipole_fraction or charge.quadrupole_fraction or charge.patch_fraction):
        return None
    lower = charge.multipole_order if charge.multipole_order < 16 else 12
    upper = min(lower + 4, 16)
    index = np.arange(64) + 0.5
    z = 1 - 2 * index / 64
    azimuth = np.pi * (3 - np.sqrt(5)) * index
    xy = np.sqrt(1 - z*z)
    unit = np.column_stack((xy * np.cos(azimuth), xy * np.sin(azimuth), z))
    points = np.concatenate((unit * 1.02 * config.geometry.radius_m,
                             unit * 2.0 * config.geometry.radius_m))
    coarse = SurfaceMultipoles(config.with_updates(charge={"multipole_order": lower}))
    fine = SurfaceMultipoles(config.with_updates(charge={"multipole_order": upper}))
    a = coarse.electric_field(points)
    b = fine.electric_field(points)
    return float(np.sqrt(np.mean((a - b)**2)) / max(np.sqrt(np.mean(b**2)), 1e-30))
