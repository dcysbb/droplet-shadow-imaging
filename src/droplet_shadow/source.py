"""有限尺寸 crossover 的电子源采样。

``phase`` 每行依次为 (x, y, x', y')；前两项为 m，后两项为
无量纲斜率 dx/dz、dy/dz，而不是以弧度存储的独立角度。
``momentum`` 每行是 (px, py, pz)，单位 kg·m/s。源位于 z=-L1；
z 坐标由两个输运器根据几何配置添加。
"""

from __future__ import annotations

import numpy as np

from .config import Source
from .constants import C, E_CHARGE, FWHM_SIGMA, M_E


def phase_space_covariance(source: Source) -> np.ndarray:
    """构造 4×4 协方差：束腰位置、发散角及 x–x'/y–y' 相关性。"""
    sx = source.crossover_fwhm_um * 1e-6 / FWHM_SIGMA
    sa = source.divergence_rms_mrad * 1e-3
    covariance = np.diag([sx**2, sx**2, sa**2, sa**2])
    covariance[0, 2] = covariance[2, 0] = source.x_xprime_correlation * sx * sa
    covariance[1, 3] = covariance[3, 1] = source.y_yprime_correlation * sx * sa
    return covariance


def geometric_emittance(source: Source) -> tuple[float, float]:
    """返回 x、y 平面的 RMS 几何发射度 sqrt(det Σ)，单位 m·rad。"""
    cov = phase_space_covariance(source)
    return tuple(float(np.sqrt(np.linalg.det(cov[np.ix_(index, index)])))
                 for index in ([0, 2], [1, 3]))


def sample_source(source: Source, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """独立抽取 n 个相空间点和动能，再转换为相对论动量。

利用 ``(pc)^2=K(K+2mc²)`` 从动能 K 得到动量模长；把斜率向量
``(x',y',1)`` 单位化后乘以模长。这样能散改变的是总动量，发散角
改变的是方向，不会重复计入能量。
"""
    phase = rng.multivariate_normal(np.zeros(4), phase_space_covariance(source), size=n)
    kinetic_j = rng.normal(source.energy_mev, source.energy_mev * source.energy_spread_rms, size=n) * 1e6 * E_CHARGE
    if np.any(kinetic_j <= 0):
        raise ValueError("Energy spread generated nonpositive electron kinetic energy")
    momentum = np.sqrt(kinetic_j * (kinetic_j + 2 * M_E * C**2)) / C
    direction = np.column_stack((phase[:, 2], phase[:, 3], np.ones(n)))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return phase, momentum[:, None] * direction
