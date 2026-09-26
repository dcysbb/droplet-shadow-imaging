"""把电子落点转换成无噪声期望图和一次合成实验曝光。

处理顺序：落点像素积分 → 按实际曝光重权 → 检测效率与 MCP 增益 →
荧光屏 PSF → 像素背景与读出噪声。``expected`` 保留平均信号，
``observed`` 只抽样一次 shot noise；不要再对它做第二次 Poisson 抽样。
"""

from __future__ import annotations

import numpy as np
import warnings
from scipy.ndimage import gaussian_filter

from .config import SimulationConfig
from .constants import FWHM_SIGMA


def image_shape(config: SimulationConfig) -> tuple[int, int]:
    """仅计算尺寸、不分配大数组；与 pixel_grid 保持相同偶数边长规则。"""
    n = int(np.ceil(config.run.raster_width_mm * 1e3 / config.detector.pixel_um))
    n += n % 2
    return n, n


def preflight_image(config: SimulationConfig) -> None:
    """在昂贵输运前提示超大视野。只警告，不偷偷改变像素或物理参数。"""
    n, _ = image_shape(config)
    if n * n > 10_000_000:
        gb = n * n * 8 / 1e9
        warnings.warn(f"Large detector image: {n} x {n} pixels; one float64 array needs "
                      f"{gb:.2f} GB and several arrays coexist. Check raster_width_mm "
                      f"({config.run.raster_width_mm}) versus detector diameter "
                      f"({config.detector.diameter_mm} mm).", RuntimeWarning, stacklevel=2)


def pixel_grid(config: SimulationConfig) -> np.ndarray:
    """返回以探测器中心对称、单位为 m 的像素边界坐标。"""
    pixel_m = config.detector.pixel_um * 1e-6
    n, _ = image_shape(config)
    return (np.arange(n + 1) - n / 2) * pixel_m


def make_images(config: SimulationConfig, hits: dict[str, np.ndarray],
                rng: np.random.Generator) -> dict[str, np.ndarray]:
    """生成 ``ideal/expected/observed`` 三层图像及像素边界。

``hits`` 可包含初级和次级电子；只排除被理想遮挡器挡住、或落在
探测器有效圆面之外的落点。理想图的值是按曝光缩放的入射数，
期望/观测图的值是 MCP 增益后的信号单位，不一定为整数。
"""
    edges = pixel_grid(config)
    detector = config.detector
    x, y = hits["x_m"], hits["y_m"]
    eligible = (~hits["blocked"]) & (x*x + y*y <= (detector.diameter_mm * 5e-4) ** 2)
    x, y = x[eligible], y[eligible]
    n_sim = config.run.n_simulated
    # 例如模拟 10^5 条而实验曝光 10^6 个，每条 MC 落点贡献平均权重 10。
    weight = config.run.exposure_electrons / n_sim
    valid_weight = np.full(len(x), weight)
    impulse, _, _ = np.histogram2d(y, x, bins=(edges, edges), weights=valid_weight)
    sigma_pixel = detector.psf_fwhm_um / FWHM_SIGMA / detector.pixel_um
    # PSF 把一个电子的光斑扩散到邻近像素，因此像素噪声会相关。
    mean_impulse = impulse * detector.efficiency * detector.gain_mean
    # 显式旁路卷积；copy 保证后续噪声处理不会反过来修改理想/期望图。
    expected = (mean_impulse.copy() if detector.psf_fwhm_um == 0 else
                gaussian_filter(mean_impulse, sigma_pixel, mode="constant"))
    # One compound-Poisson detector draw from the transport-estimated impulse
    # intensity. Sampling each weighted MC ray repeatedly at the exact same
    # coordinate would add artificial event-position clustering when weight>1.
    detected = rng.poisson(impulse * detector.efficiency)
    noisy_impulse = np.zeros_like(impulse)
    active = detected > 0
    noisy_impulse[active] = rng.gamma(detected[active] * detector.gain_shape,
                                      detector.gain_mean / detector.gain_shape)
    # Gamma 的 shape 随检测电子数增加，表示各单电子增益的独立求和。
    observed = (noisy_impulse.copy() if detector.psf_fwhm_um == 0 else
                gaussian_filter(noisy_impulse, sigma_pixel, mode="constant"))
    observed += rng.poisson(detector.background_counts_pixel, size=observed.shape)
    if detector.read_noise_rms:
        observed += rng.normal(0, detector.read_noise_rms, size=observed.shape)
    return {"ideal": impulse, "expected": expected, "observed": observed,
            "x_edges_m": edges, "y_edges_m": edges,
            "detected_fraction": np.array(len(x) / n_sim)}
