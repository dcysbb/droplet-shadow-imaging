"""无材料电子轨迹积分器：快，但只作为几何/电场验证与预览。

液滴中心为 z=0，束腰在 -L1，屏幕在 +L2。这里积分相对论
``dp/dt=-eE``，以 z 为自变量推进；不会模拟水中的散射、能损或
次级粒子，也不能可靠解析纳米界面层。正式穿滴图像应使用 Geant4。
"""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .constants import C, E_CHARGE, M_E
from .fields import Field


def z_mesh(config: SimulationConfig) -> np.ndarray:
    """构造围绕液滴加密、远处逐渐变粗的 z 网格（m）。"""
    g, run = config.geometry, config.run
    R = g.radius_m
    min_z = min(run.transport_step_um * 1e-6, R / 12)
    start = g.l1_mm * 1e-3
    end = g.l2_mm * 1e-3
    # A geometric mesh resolves the droplet and still integrates the long
    # Coulomb tail up to the detector. Molecular layers require Geant4.
    before = np.geomspace(min_z, start, max(60, int(np.log(start / min_z) * 45)))
    after = np.geomspace(min_z, end, max(60, int(np.log(end / min_z) * 45)))
    return np.r_[-before[::-1], 0.0, after]


def trace_rays(config: SimulationConfig, field: Field, phase: np.ndarray,
               momentum: np.ndarray) -> dict[str, np.ndarray]:
    """逐个 z 小段推进全部电子，返回与 Geant4 相同语义的落点字典。

``ideal_occluder`` 只额外标记几何上会碰到球的粒子；其余轨迹
仍按给定场推进。``entered`` 表示网格中有采样点位于球内，
不是 Geant4 几何边界的精确穿越判定。
"""
    n = len(phase)
    x = phase[:, 0].copy()
    y = phase[:, 1].copy()
    px, py, pz = momentum[:, 0].copy(), momentum[:, 1].copy(), momentum[:, 2].copy()
    if config.run.engine == "ideal_occluder":
        slope2 = phase[:, 2] ** 2 + phase[:, 3] ** 2
        r_closest = np.sqrt((x + config.geometry.l1_mm * 1e-3 * phase[:, 2]) ** 2 +
                            (y + config.geometry.l1_mm * 1e-3 * phase[:, 3]) ** 2)
        blocked = r_closest <= config.geometry.radius_m * np.sqrt(1 + slope2)
    else:
        blocked = np.zeros(n, dtype=bool)
    entered = np.zeros(n, dtype=bool)
    for z0, z1 in zip(z_mesh(config)[:-1], z_mesh(config)[1:]):
        dz = z1 - z0
        xmid = x + 0.5 * dz * px / pz
        ymid = y + 0.5 * dz * py / pz
        zmid = 0.5 * (z0 + z1)
        xyz = np.column_stack((xmid, ymid, np.full(n, zmid)))
        entered |= np.einsum("ij,ij->i", xyz, xyz) < config.geometry.radius_m**2
        electric = field.electric_field(xyz)
        # p=γmv，所以 dz/vz 是近似经过这个 z 小段的时间 dt。
        gamma_m = np.sqrt(M_E**2 + (px**2 + py**2 + pz**2) / C**2)
        vz = pz / gamma_m
        px += -E_CHARGE * electric[:, 0] * dz / vz
        py += -E_CHARGE * electric[:, 1] * dz / vz
        pz += -E_CHARGE * electric[:, 2] * dz / vz
        x += dz * px / pz
        y += dz * py / pz
    p2 = px**2 + py**2 + pz**2
    energy_mev = (np.sqrt(M_E**2 * C**4 + p2 * C**2) - M_E * C**2) / (E_CHARGE * 1e6)
    return {"x_m": x, "y_m": y, "energy_mev": energy_mev,
            "entered": entered, "blocked": blocked,
            "primary": np.ones(n, dtype=bool), "event_id": np.arange(n)}


def weak_deflection_angle(q_e: float, impact_m: float, energy_mev: float) -> float:
    """直线弱偏转解析解：电子从无限远掠过点电荷后的有符号角度。

正液滴吸引负电子，故正 ``q_e`` 的偏转角为负。公式只适用于
冲量近似与足够大的入射参数；用来核对数值轨迹的符号和量级。
"""
    from .constants import COULOMB_K
    K = energy_mev * 1e6 * E_CHARGE
    pv_j = K * (K + 2 * M_E * C**2) / (K + M_E * C**2)
    return -2 * COULOMB_K * q_e * E_CHARGE**2 / (impact_m * pv_j)
