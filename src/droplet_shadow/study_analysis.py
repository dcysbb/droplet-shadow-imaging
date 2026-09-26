"""只读输运结果的人工检查工具；本模块绝不调用 simulate/run_geant4。

期望图仍有有限输运采样纹理。曝光噪声只从原始落点重新生成一次，
从不在 observed 上二次采样。图像不做额外平滑，不执行自动可见性判断。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .analysis import radial_profile
from .config import SimulationConfig
from .detector import make_images
from .fields import build_field
from .study import EXPOSURES, StudyCase, atomic_json, completed_path, sha256

COLORS = {10: "#214e7a", 5: "#b96920", 1: "#687b36"}


def save_figure(fig, path: Path) -> None:
    """同时保存便于快速查看的 PNG 和可编辑的 SVG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".svg"))


def field_figure(case: StudyCase):
    """单模型全径向范围及界面局部；全部线性坐标，显示有符号场。"""
    cfg = case.config
    R = cfg.geometry.radius_m
    charge = cfg.charge
    field = build_field(cfg)
    molecular = bool(charge.dipole_potential_v)
    shell = charge.model == "neutral_double_layer"
    span = (max(5.0, charge.dipole_thickness_nm * 3) * 1e-9 if molecular else
            max(20.0, charge.double_layer_thickness_nm * 3) * 1e-9 if shell else 10e-6)
    if charge.model == "poisson_boltzmann":
        # 浓度不同的 PB 层可相差两个数量级。缩放到约五个半衰减深度，
        # 避免 1 μM 层在 ±10 μm 的“放大图”里仍只剩一根竖线。
        # 这只是显示窗口，不改变解、取样积分或声称它是 Debye 长度。
        magnitude = np.abs(field.field_v_m)
        if magnitude.max() > 0:
            first_half = field.radii_m[np.flatnonzero(magnitude >= .5 * magnitude.max())[0]]
            span = float(np.clip(5 * (R - first_half), 3e-9, 10e-6))
    edges = [R]
    if molecular:
        edges.append(R - charge.dipole_thickness_nm * 1e-9)
    if shell:
        edges.append(R - charge.double_layer_thickness_nm * 1e-9)
    # 跳变两侧显式采样，不能靠稀疏均匀网格偶然碰到亚纳米层。
    special = np.r_[edges, np.nextafter(edges, 0.0)]
    radii_full = np.unique(np.r_[np.linspace(0, 4 * R, 1800), special])
    radii_zoom = np.unique(np.r_[np.linspace(R - span, R + span, 2400),
                                 R - np.geomspace(1e-12, span, 800), special])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained")
    scale = 1e9 if molecular or shell else 1e6
    zoom_unit = "nm" if scale == 1e9 else "μm"
    for column, radii in enumerate((radii_full, radii_zoom)):
        xyz = np.column_stack((radii, np.zeros_like(radii), np.zeros_like(radii)))
        x = radii * 1e6 if column == 0 else (radii - R) * scale
        axes[0, column].plot(x, field.electric_field(xyz)[:, 0] / 1e6, color="#214e7a")
        axes[1, column].plot(x, field.potential(xyz), color="#b96920")
        for ax in axes[:, column]:
            ax.axvline(R * 1e6 if column == 0 else 0, color="0.45", ls="--", lw=.8)
            ax.set_xlabel("Radius (μm)" if column == 0 else f"r − R ({zoom_unit})")
            ax.grid(alpha=.15)
        axes[0, column].set_ylabel("Radial E (MV/m)")
        axes[1, column].set_ylabel("Potential (V)")
    fig.suptitle(f"Model field only — {case.model}; R={cfg.geometry.radius_um:g} μm; Q={charge.q_e:g} e")
    return fig


def classified_profiles(config: SimulationConfig, hits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """原始落点径向统计：等宽环、绝对粒子数、十个初级事件子样本。

这里按真实落点半径分箱，不把像素中心近似当作电子真实位置。密度
单位为“每个入射初级电子、每 mm² 的落点数”，不是归一化概率密度；
有限探测器造成的通量损失保留。次级电子与其母事件在同一子样本，
没有落点的初级电子仍计入分母。子样本曲线不是置信区间。
"""
    n = config.run.n_simulated
    if n < 10:
        raise ValueError("Ten event subsets require at least ten primary events")
    ids = np.asarray(hits["event_id"])
    if ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids >= n)):
        raise ValueError("Invalid event IDs")
    radius = np.hypot(hits["x_m"], hits["y_m"])
    limit = config.detector.diameter_mm * .5e-3
    pixel = config.detector.pixel_um * 1e-6
    # 整数个像素的半径可能因 SI 浮点换算变为 800.0000000000001；
    # 不能 ceil 成 801 再把末端压回 R，否则产生零面积环和 NaN。
    ratio = limit / pixel
    bins = int(round(ratio) if np.isclose(ratio, round(ratio), rtol=0, atol=1e-9) else np.ceil(ratio))
    edges = np.arange(bins + 1) * pixel
    edges[-1] = limit
    area_mm2 = np.pi * np.diff(edges**2) * 1e6
    eligible = (~hits["blocked"]) & (radius <= limit)
    categories = {
        "all": eligible,
        "primary_unentered": eligible & hits["primary"] & ~hits["entered"],
        "primary_entered": eligible & hits["primary"] & hits["entered"],
        "secondary": eligible & ~hits["primary"],
    }
    result = {"radius_detector_m": (edges[:-1] + edges[1:]) / 2,
              "radius_object_um": (edges[:-1] + edges[1:]) * .5e6 / config.geometry.magnification,
              "annulus_area_mm2": area_mm2}
    for name, selected in categories.items():
        counts = np.histogram(radius[selected], edges)[0]
        result[name + "_counts"] = counts
        result[name + "_per_primary_mm2"] = counts / (n * area_mm2)
    boundaries = np.linspace(0, n, 11, dtype=np.int64)
    subset_counts = np.stack([np.histogram(radius[eligible & (ids >= lo) & (ids < hi)], edges)[0]
                              for lo, hi in zip(boundaries[:-1], boundaries[1:])])
    result["subset_primary_counts"] = np.diff(boundaries)
    result["subset_counts"] = subset_counts
    result["subset_per_primary_mm2"] = subset_counts / np.diff(boundaries)[:, None] / area_mm2
    return result


def profile_figure(profiles: dict, case: StudyCase):
    """分类绝对通量与十个子样本波动；不寻找环、不拟合阈值。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    r = profiles["radius_object_um"]
    styles = (("all", "All hits", "#214e7a", "-"),
              ("primary_unentered", "Primary, did not enter water", "#687b36", "--"),
              ("primary_entered", "Primary, entered water", "#b96920", "-."),
              ("secondary", "Secondary", "#a44f79", ":"))
    for key, title, color, style in styles:
        axes[0].plot(r, profiles[key + "_per_primary_mm2"], label=title, color=color, ls=style)
    for index, profile in enumerate(profiles["subset_per_primary_mm2"]):
        axes[1].plot(r, profile, color="#214e7a", alpha=.3, lw=.8,
                     label="10 disjoint event subsets" if index == 0 else None)
    axes[1].plot(r, profiles["all_per_primary_mm2"], color="0.2", label="All-event mean", lw=1.3)
    for ax in axes:
        ax.set(xlabel="Object-equivalent radius (μm)", ylabel="Hits / incident primary / detector mm²")
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    width = case.config.detector.pixel_um / case.config.geometry.magnification
    fig.suptitle(f"{case.case_id} — raw hits; N={case.config.run.n_simulated:,}; object radial bin={width:.4g} μm")
    return fig


def _image_figure(images: dict, case: StudyCase, exposure: int):
    edges = images["x_edges_m"] * 1e3
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    maximum = max(float(np.max(images[key])) for key in ("expected", "observed")) or 1
    for ax, key in zip(axes, ("expected", "observed")):
        shown = ax.imshow(images[key], origin="lower", extent=extent, cmap="magma",
                          vmin=0, vmax=maximum, interpolation="nearest")
        ax.set(xlabel="Detector x (mm)", ylabel="Detector y (mm)", title=key)
    fig.colorbar(shown, ax=axes, label="Signal / pixel (gain units)")
    fig.suptitle(f"{case.case_id}; incident exposure={exposure:,}; PSF=0; no display blur")
    return fig


def analyze_case(case: StudyCase, attempt: Path, destination: Path) -> None:
    """后处理一个已完成目录，三个曝光共享输运落点，不再追踪电子。"""
    if case.config.detector.psf_fwhm_um != 0:
        raise ValueError("This study requires PSF=0")
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(attempt / "hits.npz", allow_pickle=False) as archive:
        hits = {key: archive[key] for key in archive.files}
    profiles = classified_profiles(case.config, hits)
    np.savez_compressed(destination / "profiles.npz", **profiles)
    with (destination / "profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = [key for key, value in profiles.items() if value.ndim == 1 and key != "subset_primary_counts"]
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*(profiles[key] for key in columns)))
    fig = profile_figure(profiles, case)
    save_figure(fig, destination / "classified_profiles")
    plt.close(fig)
    exposure_report = {}
    case_seed = int(hashlib.sha256(case.case_id.encode()).hexdigest()[:8], 16)
    for exposure in EXPOSURES:
        cfg = case.config.with_updates(run={"exposure_electrons": exposure})
        rng = np.random.default_rng(np.random.SeedSequence([cfg.run.seed, exposure, case_seed]))
        images = make_images(cfg, hits, rng)
        np.savez_compressed(destination / f"exposure_{exposure}.npz", **images)
        fig = _image_figure(images, case, exposure)
        save_figure(fig, destination / f"exposure_{exposure}")
        plt.close(fig)
        charge_change = float(np.sum(hits["deposited_e_by_event"])) * exposure / cfg.run.n_simulated
        exposure_report[str(exposure)] = {
            "expected_signal_sum": float(images["expected"].sum()),
            "observed_signal_sum": float(images["observed"].sum()),
            "projected_charge_change_e": charge_change,
            "projected_abs_charge_change_over_abs_Q": abs(charge_change / cfg.charge.q_e) if cfg.charge.q_e else None}
    atomic_json(destination / "analysis.json", {
        "case_id": case.case_id, "config": case.config.to_dict(),
        "input_hits_sha256": sha256(attempt / "hits.npz"),
        "analysis_code_sha256": sha256(Path(__file__)), "exposures": exposure_report,
        "judgment": "Not classified. Inspect images and event-subset variation manually.",
        "caveat": "Expected images retain finite transport-MC texture; subsets are not confidence intervals."})


def read_exposure(directory: Path, exposure: int = 1_000_000) -> dict:
    with np.load(directory / f"exposure_{exposure}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compatible_reference(case: StudyCase, reference: StudyCase) -> None:
    """几何/源/探测器及入射采样必须匹配；只允许电场及水材料不同。"""
    a, b = case.config, reference.config
    if (a.geometry != b.geometry or a.source != b.source or a.detector != b.detector or
            a.run.n_simulated != b.run.n_simulated or a.run.seed != b.run.seed):
        raise ValueError(f"Incompatible reference: {case.case_id}, {reference.case_id}")


def comparison_figures(case: StudyCase, neutral: StudyCase, absent: StudyCase, analysis: Path):
    """全视野/固定物方边缘窗口/像素径向曲线；差分永远不额外平滑。"""
    compatible_reference(case, neutral)
    compatible_reference(case, absent)
    images = [read_exposure(analysis / c.case_id) for c in (case, neutral, absent)]
    for other in images[1:]:
        if not np.array_equal(images[0]["x_edges_m"], other["x_edges_m"]):
            raise ValueError("Comparison pixel grids differ")
    arrays = [item["expected"] for item in images]
    diff = arrays[0] - arrays[1]
    edges = images[0]["x_edges_m"] * 1e3
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    vmax = max(float(np.max(a)) for a in arrays) or 1
    dmax = float(np.max(abs(diff))) or 1e-12
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    M = case.config.geometry.magnification
    for column, (array, label) in enumerate(zip([*arrays, diff], [case.model, "No-field water", "No droplet", "Model − no-field water"])):
        for row in range(2):
            ax = axes[row, column]
            shown = ax.imshow(array, origin="lower", extent=extent, interpolation="nearest",
                              cmap="RdBu_r" if column == 3 else "magma",
                              vmin=-dmax if column == 3 else 0, vmax=dmax if column == 3 else vmax)
            ax.set(xlabel="Detector x (mm)", ylabel="Detector y (mm)", title=label)
            if row == 1:
                # 固定物方窗口，避免小焦点组被不一致的自动缩放夸大。
                R_um = case.config.geometry.radius_um
                ax.set_xlim((R_um - 20) * M / 1000, (R_um + 20) * M / 1000)
                ax.set_ylim(-20 * M / 1000, 20 * M / 1000)
            fig.colorbar(shown, ax=ax, label="Signal / pixel")
    fig.suptitle(f"{case.case_id}; exposure=1,000,000; PSF=0; absolute flux; no display blur")
    radial, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    bins = int(round(case.config.detector.diameter_mm * 1000 / 2 / case.config.detector.pixel_um))
    for item, label, color, style in zip(images, [case.model, "No-field water", "No droplet"],
                                        ["#214e7a", "#b96920", "0.35"], ["-", "--", ":"]):
        r, profile = radial_profile(item["expected"], item["x_edges_m"], bins=bins)
        ax.plot(r * 1e6 / M, profile, label=label, color=color, ls=style)
    ax.set(xlabel="Object-equivalent radius (μm)", ylabel="Mean expected signal / pixel",
           title=f"{case.case_id}; radial bin={case.config.detector.pixel_um:g} μm detector / "
                 f"{case.config.detector.pixel_um / M:.4g} μm object")
    ax.legend()
    return fig, radial


def focus_figure(cases: list[StudyCase], analysis: Path, *, edge_zoom: bool = False):
    """相同模型/距离的三个源宽度，共用强度色标与差分色标。"""
    cases = sorted(cases, key=lambda c: c.config.source.crossover_fwhm_um, reverse=True)
    if len(cases) != 3 or len({(c.model, c.config.geometry) for c in cases}) != 1:
        raise ValueError("Focus comparison requires three widths of the same model and geometry")
    data = [read_exposure(analysis / c.case_id) for c in cases]
    refs = [read_exposure(analysis / c.neutral_case) for c in cases]
    arrays = [d["expected"] for d in data]
    differences = [a - r["expected"] for a, r in zip(arrays, refs)]
    vmax = max(float(np.max(a)) for a in arrays) or 1
    dmax = max(float(np.max(abs(a))) for a in differences) or 1e-12
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for i, case in enumerate(cases):
        edge = data[i]["x_edges_m"] * 1e3
        extent = [edge[0], edge[-1], edge[0], edge[-1]]
        for row, array in enumerate((arrays[i], differences[i])):
            shown = axes[row, i].imshow(array, origin="lower", extent=extent, interpolation="nearest",
                cmap="magma" if row == 0 else "RdBu_r", vmin=0 if row == 0 else -dmax,
                vmax=vmax if row == 0 else dmax)
            axes[row, i].set(title=f"Source FWHM={case.config.source.crossover_fwhm_um:g} μm" +
                            ("" if row == 0 else "; model − no-field water"),
                            xlabel="Detector x (mm)", ylabel="Detector y (mm)")
            if edge_zoom:
                M = case.config.geometry.magnification
                R = case.config.geometry.radius_um
                axes[row, i].set_xlim((R - 20) * M / 1000, (R + 20) * M / 1000)
                axes[row, i].set_ylim(-20 * M / 1000, 20 * M / 1000)
            fig.colorbar(shown, ax=axes[row, i], label="Signal / pixel")
    fig.suptitle(f"{cases[0].model}; exposure=1,000,000; fixed 10 mrad RMS; PSF=0")
    return fig


def focus_radial_figure(cases: list[StudyCase], analysis: Path):
    """三焦点绝对径向强度及各自减无场水滴的曲线，不拟合、不额外平滑。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for case in sorted(cases, key=lambda c: c.config.source.crossover_fwhm_um, reverse=True):
        data = read_exposure(analysis / case.case_id)
        ref = read_exposure(analysis / case.neutral_case)
        if not np.array_equal(data["x_edges_m"], ref["x_edges_m"]):
            raise ValueError("Focus reference pixel grids differ")
        cfg = case.config
        bins = round(cfg.detector.diameter_mm * 1000 / 2 / cfg.detector.pixel_um)
        focus = cfg.source.crossover_fwhm_um
        for ax, array in zip(axes, (data["expected"], data["expected"] - ref["expected"])):
            r, profile = radial_profile(array, data["x_edges_m"], bins=bins)
            ax.plot(r * 1e6 / cfg.geometry.magnification, profile, color=COLORS.get(focus),
                    label=f"Source FWHM={focus:g} μm")
    for ax, title in zip(axes, ("Absolute expected signal", "Model − matched no-field water")):
        ax.set(xlabel="Object-equivalent radius (μm)", ylabel="Mean expected signal / pixel", title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    cfg = cases[0].config
    fig.suptitle(f"{cases[0].model}; exposure=1,000,000; object radial bin="
                 f"{cfg.detector.pixel_um / cfg.geometry.magnification:.4g} μm; no extra smoothing")
    return fig


def analyze_study(cases: list[StudyCase], output: Path, *, selected: set[str] | None = None) -> dict:
    """仅处理已有、校验成功的结果；缺失结果保持 missing，不补跑输运。"""
    analysis = Path(output) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    ready = {}
    status = {}
    for case in cases:
        if selected is not None and case.case_id not in selected:
            continue
        attempt = completed_path(case, output)
        if attempt is None:
            status[case.case_id] = "missing — no transport invoked"
            continue
        analyze_case(case, attempt, analysis / case.case_id)
        ready[case.case_id] = case
        status[case.case_id] = "analyzed"
    # 只有本次校验并分析的匹配参照才能参与比较，避免混入旧派生文件。
    for case in ready.values():
        if case.neutral_case in ready and case.absent_case in ready:
            figs = comparison_figures(case, ready[case.neutral_case], ready[case.absent_case], analysis)
            for fig, label in zip(figs, ("comparison", "radial_comparison")):
                save_figure(fig, analysis / case.case_id / label)
                plt.close(fig)
    for model in sorted({c.model for c in ready.values()}):
        group = [c for c in ready.values() if c.model == model and c.group == "main"]
        if len(group) == 3 and all(c.neutral_case in ready for c in group):
            for fig, suffix in ((focus_figure(group, analysis), ""),
                                (focus_figure(group, analysis, edge_zoom=True), "_edge"),
                                (focus_radial_figure(group, analysis), "_radial")):
                save_figure(fig, analysis / "focus" / (model + suffix))
                plt.close(fig)
    atomic_json(analysis / "status.json", status)
    return status
