"""从图像提取可观测量，并在给定仪器情景下拟合电荷。

半径、亮/暗环只有在图像满足相应定义时才给数值；``None`` 表示
不可定义。电荷拟合比较同几何、同源、同探测器的模板，统计分数
必须再用独立伪实验校准，不能直接当作普适置信度或检出限。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter, shift as image_shift
from scipy.optimize import minimize

from .config import SimulationConfig
from .constants import FWHM_SIGMA


def spatial_resolution_object_um(config: SimulationConfig) -> float:
    """把有限束腰、PSF、像素积分的 RMS 宽度合成，再报物面 FWHM。

先用高斯换算 ``FWHM/2.355=σ``；像素均匀积分的 ``σ=p/√12``。
物面源项受 ``(M-1)/M`` 缩放，屏幕项除以 M。这里只是几何/探测器
分辨率，不包含穿水滴后的多重散射导致的对比度损失。
"""
    M = config.geometry.magnification
    sigma_source = config.source.crossover_fwhm_um / FWHM_SIGMA
    sigma_psf = config.detector.psf_fwhm_um / FWHM_SIGMA
    sigma_pixel = config.detector.pixel_um / math.sqrt(12)
    return math.sqrt((sigma_source * (M - 1) / M) ** 2 +
                     (sigma_psf / M) ** 2 + (sigma_pixel / M) ** 2) * FWHM_SIGMA


def radial_profile(image: np.ndarray, edges_m: np.ndarray, bins: int = 100,
                   center_m: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """按探测器半径把像素划入同心环，返回环中心 m 和平均像素信号。"""
    centers = 0.5 * (edges_m[:-1] + edges_m[1:])
    xx, yy = np.meshgrid(centers - center_m[0], centers - center_m[1])
    radius = np.hypot(xx, yy)
    limit = min(abs(edges_m[0]), abs(edges_m[-1]))
    boundaries = np.linspace(0, limit, bins + 1)
    index = np.digitize(radius.ravel(), boundaries) - 1
    good = (index >= 0) & (index < bins)
    sums = np.bincount(index[good], weights=image.ravel()[good], minlength=bins)
    counts = np.bincount(index[good], minlength=bins)
    return 0.5 * (boundaries[:-1] + boundaries[1:]), sums / np.maximum(counts, 1)


def image_metrics(config: SimulationConfig, image: np.ndarray,
                  reference: np.ndarray, edges_m: np.ndarray) -> dict[str, float | None]:
    """量化 shadow 相对参考图的几何边界、环和绝对通量变化。

参考图应为同条件中性滴或无滴图，且不得事先分别归一化通量。
``shadow_radius_um`` 使用相对强度 0.5 的上穿越半深度边界；
没有可靠穿越就返回 ``None``。亮/暗环也要求可靠环带里有足够
输运样本，并达到 ±2% 对比度，避免把 MC 空环当成物理焦散。
"""
    r, profile = radial_profile(image, edges_m)
    _, ref = radial_profile(reference, edges_m)
    R_proj = config.geometry.radius_m * config.geometry.magnification
    within = r < 1.5 * R_proj
    deficit = float(np.sum(np.maximum(ref[within] - profile[within], 0)))
    ratio = np.divide(profile, ref, out=np.ones_like(profile), where=ref > 1e-9)
    centers = 0.5 * (edges_m[:-1] + edges_m[1:])
    xx, yy = np.meshgrid(centers, centers)
    radius = np.hypot(xx, yy)
    boundaries = np.linspace(0, min(abs(edges_m[0]), abs(edges_m[-1])), len(r) + 1)
    annulus = np.digitize(radius.ravel(), boundaries) - 1
    valid = (annulus >= 0) & (annulus < len(r))
    ref_annulus = np.bincount(annulus[valid], weights=reference.ravel()[valid],
                             minlength=len(r))
    per_simulated_electron = (config.run.exposure_electrons / config.run.n_simulated *
                              config.detector.efficiency * config.detector.gain_mean)
    # 从参考环总信号反推该环的输运历史数，过滤粒子过少的环。
    reference_mc_events = ref_annulus / max(per_simulated_electron, 1e-12)
    # A sparse transport template can create spurious giant ratios. Its ring
    # extrema are not a measurement until an annulus has enough source events.
    reliable = reference_mc_events >= 50
    mask = (r > 0.6 * R_proj) & (r < 1.7 * R_proj) & reliable
    if mask.any():
        rim = np.where(mask)[0]
        dark_idx = rim[np.argmin(ratio[rim])]
        bright_idx = rim[np.argmax(ratio[rim])]
        dark = float(ratio[dark_idx]) if ratio[dark_idx] < 0.98 else None
        bright = float(ratio[bright_idx]) if ratio[bright_idx] > 1.02 else None
    else:
        dark_idx = bright_idx = 0
        dark = bright = None
    # A radius is defined only when a half-depth contour exists.
    edge_idx = np.where((r[:-1] < 1.5 * R_proj) & reliable[:-1] & reliable[1:] &
                        (ratio[:-1] < 0.5) & (ratio[1:] >= 0.5))[0]
    radius_um = float(r[edge_idx[-1]] / config.geometry.magnification * 1e6) if edge_idx.size else None
    inner_pixels = radius < 1.5 * R_proj
    absolute_depletion = float(np.sum(np.maximum(reference[inner_pixels] - image[inner_pixels], 0)))
    signed_difference = float(np.sum(reference[inner_pixels] - image[inner_pixels]))
    return {"shadow_radius_um": radius_um,
            "dark_ring_ratio": dark, "dark_ring_r_mm": float(r[dark_idx] * 1e3) if dark else None,
            "bright_ring_ratio": bright, "bright_ring_r_mm": float(r[bright_idx] * 1e3) if bright else None,
            "depletion_proxy": deficit,
            "depletion_signal_units": absolute_depletion,
            "signed_inner_flux_difference_signal_units": signed_difference,
            "total_image_signal_units": float(np.sum(image)),
            "total_reference_signal_units": float(np.sum(reference)),
            "resolution_fwhm_object_um": spatial_resolution_object_um(config)}


def angular_harmonics(image: np.ndarray, edges_m: np.ndarray,
                      inner_m: float, outer_m: float, max_order: int = 4) -> dict[int, float]:
    """环带上计算 ``|Σ I exp(ikφ)|/ΣI``，测量图像的角向不对称。

``k=1`` 近似左右偏斜，``k=2`` 近似双瓣；这只是二维投影描述量，
单次投影一般无法唯一恢复三维表面电荷。
"""
    centers = (edges_m[:-1] + edges_m[1:]) * 0.5
    x, y = np.meshgrid(centers, centers)
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    zone = (r >= inner_m) & (r < outer_m)
    weights = image[zone]
    total = np.sum(weights)
    if total <= 0:
        return {k: float("nan") for k in range(1, max_order + 1)}
    return {k: float(abs(np.sum(weights * np.exp(1j * k * a[zone]))) / total)
            for k in range(1, max_order + 1)}


def poisson_deviance(observed: np.ndarray, expectation: np.ndarray,
                     variance_extra: float = 0.0) -> float:
    """计算带均值相关方差的加权平方残差，越小表示模板越符合图像。

这是复合 Poisson/MCP 增益信号的近似分数，不是精确解析似然。
必须用独立、逐事件模拟曝光来标定判别阈值。
"""
    variance = np.maximum(expectation, 1.0) + variance_extra
    residual = observed - expectation
    return float(np.sum(residual * residual / variance))


@dataclass
class ChargeFit:
    """离散 Q 模板网格上的最佳电荷、最小分数和全部分数。"""

    q_e: float
    score: float
    scores: dict[float, float]


def fit_charge(observed: np.ndarray, templates: dict[float, np.ndarray],
               read_noise_rms: float = 0.0,
               fit_nuisance: bool = True) -> ChargeFit:
    """逐个电荷模板拟合图像，并对非电荷干扰参数进行轮廓优化。

干扰量包括正通量倍率、非负背景、x/y 中心偏移和额外模糊；
这些量可能与 Q 部分简并，所以单次最小分数不等于“已检出电荷”。
读出噪声通过额外方差进入分数。模板本身应先由更高统计量产生。
"""
    scores = {}
    for q, expected in templates.items():
        if expected.shape != observed.shape:
            raise ValueError("Template shape does not match observed image")
        # Fit positive flux and additive background on a bounded central crop.
        size = min(expected.shape)
        width = min(size // 2, 180)
        cy, cx = np.array(expected.shape) // 2
        e = expected[cy-width:cy+width, cx-width:cx+width]
        o = observed[cy-width:cy+width, cx-width:cx+width]
        A = np.column_stack((e.ravel(), np.ones(e.size)))
        scale, background = np.linalg.lstsq(A, o.ravel(), rcond=None)[0]
        if fit_nuisance:
            # Position and excess source/PSF width are scenario nuisance
            # parameters. Thresholds must still be calibrated by pseudodata.
            def objective(values: np.ndarray) -> float:
                # 每个候选 Q 都要同等允许位置、亮度与额外 PSF 调整。
                flux, bg, dy, dx, blur = values
                shifted = image_shift(e, (dy, dx), order=1, mode="nearest")
                if blur > 0.05:
                    shifted = gaussian_filter(shifted, blur, mode="nearest")
                pred = np.maximum(1e-9, flux * shifted + bg)
                return poisson_deviance(o, pred, read_noise_rms**2)
            solution = minimize(objective,
                                np.array([max(scale, 0), max(background, 0), 0, 0, 0]),
                                method="L-BFGS-B",
                                bounds=[(0, None), (0, None), (-2, 2), (-2, 2), (0, 3)],
                                options={"maxiter": 35, "ftol": 1e-5})
            scores[float(q)] = float(solution.fun)
        else:
            prediction = np.maximum(1e-9, max(scale, 0.0) * e + max(background, 0.0))
            scores[float(q)] = poisson_deviance(o, prediction, read_noise_rms**2)
    best = min(scores, key=scores.get)
    return ChargeFit(best, scores[best], scores)


def calibrated_detection_limit(null_scores: np.ndarray,
                               alternative_scores: dict[float, np.ndarray],
                               false_positive: float = 0.05,
                               power: float = 0.95) -> float | None:
    """返回通过零电荷阈值与 95% 检出率的最小*已测试* |Q|。

若任何备择都达不到要求则返回 ``None``，表示当前网格中未检出，
不是零电荷，也不是对所有更小 Q 的数学证明。
"""
    threshold = float(np.quantile(null_scores, 1 - false_positive, method="higher"))
    detected = [abs(q) for q, values in alternative_scores.items()
                if q and np.mean(np.asarray(values) > threshold) >= power]
    return min(detected) if detected else None


def caustic_locations(config: SimulationConfig, field, count: int = 120) -> list[float]:
    """寻找理想无散射映射的径向 Jacobian 变号位置，单位物面 μm。

轴对称映射的面积 Jacobian 与 ``(dr_D/db)·(r_D/b)`` 成正比；
零点才是 caustic 候选。有限源尺寸和材料散射可能把候选完全抹平，
所以普通图像亮环不能仅凭外观称为 caustic。
"""
    from .constants import C, E_CHARGE, M_E
    from .rays import trace_rays
    R = config.geometry.radius_m
    impact = np.geomspace(R * 1.02, R * 8, count)
    phase = np.column_stack((np.zeros(count), np.zeros(count),
                             impact / (config.geometry.l1_mm * 1e-3), np.zeros(count)))
    K = config.source.energy_mev * 1e6 * E_CHARGE
    p = np.sqrt(K * (K + 2 * M_E * C**2)) / C
    slopes = phase[:, 2]
    momentum = np.column_stack((p * slopes / np.sqrt(1 + slopes**2),
                                np.zeros(count), p / np.sqrt(1 + slopes**2)))
    ray_cfg = config.with_updates(run={"engine": "ray", "n_simulated": count,
                                       "exposure_electrons": count})
    result = trace_rays(ray_cfg, field, phase, momentum)
    mapped = result["x_m"]
    determinant = np.gradient(mapped, impact) * mapped / impact
    crossing = np.where(np.signbit(determinant[:-1]) != np.signbit(determinant[1:]))[0]
    return [float(impact[i] * 1e6) for i in crossing]
