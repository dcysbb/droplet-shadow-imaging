"""通过独立合成曝光校准“可检测净电荷”的统计判据。

输入是同条件但更高输运统计量的期望图模板。每个候选 Q 重新抽样
电子源、完整输运和探测器，前半组估计阈值，后半组只做验证，
避免在同一批随机数据上既定阈值又报告检出率。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .analysis import calibrated_detection_limit, fit_charge
from .config import SimulationConfig
from .simulation import load_images, simulate


@dataclass(frozen=True)
class DetectionCalibration:
    """一个固定仪器情景和离散 Q 网格下的校准与留出验证摘要。"""

    threshold: float
    false_positive_rate: float
    power_by_charge: dict[float, float]
    minimum_detectable_abs_q_e: float | None
    repetitions: int
    calibration_repetitions: int
    validation_repetitions: int
    charge_bias_by_true_q_e: dict[float, float]
    interval_coverage_by_true_q_e: dict[float, float]


def detection_calibration(config: SimulationConfig,
                          template_directories: dict[float, str | Path],
                          output_dir: str | Path, repetitions: int = 100,
                          fit_nuisance: bool = True) -> DetectionCalibration:
    """为每个 Q 生成独立的源+输运+探测器曝光，评估 5%/95% 判据。

模板必须单独以更高输运统计量制作。所得阈值依赖模板 Q 网格、
光源、探测器、曝光、静态液滴假设，不是仪器无关的普适检出限。
``repetitions`` 对每个 Q 分配相同次数，至少 4 次仅够测试流程；
估计 5% 尾概率实际应使用远多于 4 次的独立曝光。
"""
    if repetitions < 4:
        raise ValueError("At least four repetitions are required for calibration/validation splitting")
    if 0.0 not in template_directories or len(template_directories) < 2:
        raise ValueError("Templates must include Q=0 and at least one nonzero Q")
    if config.charge.model == "poisson_boltzmann":
        raise ValueError("PB charge calibration requires per-Q ion populations; use explicit coupled configs")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    templates: dict[float, np.ndarray] = {}
    for q, directory in template_directories.items():
        # 模板的所有干扰条件必须匹配；否则图像差异可能不是 Q 引起的。
        metadata = json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))
        source = metadata["config"]
        for section in ("geometry", "source", "detector"):
            if source[section] != config.to_dict()[section]:
                raise ValueError(f"Template Q={q:g} has different {section} settings")
        if float(source["charge"]["q_e"]) != float(q):
            raise ValueError(f"Template directory for Q={q:g} has the wrong charge")
        expected_charge = config.to_dict()["charge"].copy()
        template_charge = source["charge"].copy()
        expected_charge.pop("q_e")
        template_charge.pop("q_e")
        if template_charge != expected_charge:
            raise ValueError(f"Template Q={q:g} uses another interface-charge model")
        if source["run"]["engine"] != config.run.engine:
            raise ValueError(f"Template Q={q:g} uses another transport engine")
        run_comparables = ("droplet_material", "geant4_world_step_mm", "geant4_far_step_mm",
                           "geant4_near_step_um", "geant4_core_step_um", "geant4_cut_um",
                           "geant4_layer_step_scale")
        for parameter in run_comparables:
            if source["run"].get(parameter, getattr(config.run, parameter)) != getattr(config.run, parameter):
                raise ValueError(f"Template Q={q:g} has another {parameter}")
        templates[float(q)] = load_images(directory)["expected"]
    scores: dict[float, list[float]] = {q: [] for q in templates}
    all_fit_scores: dict[float, list[dict[float, float]]] = {q: [] for q in templates}
    fitted_charges: dict[float, list[float]] = {q: [] for q in templates}
    calibration_count = repetitions // 2
    records = []
    for q in sorted(templates):
        for repeat in range(repetitions):
            # 不同 Q 和不同重复编号都使用独立种子，避免重复同一电子束。
            case = config.with_updates(charge={"q_e": q},
                                       run={"seed": config.run.seed + repeat +
                                            100003 * (1 + list(sorted(templates)).index(q))})
            run_dir = output / f"Q_{q:+.0f}" / f"repeat_{repeat:04d}"
            observed = simulate(case, run_dir).images["observed"]
            fit = fit_charge(observed, templates,
                             read_noise_rms=config.detector.read_noise_rms,
                             fit_nuisance=fit_nuisance)
            statistic = fit.scores[0.0] - min(value for charge, value in fit.scores.items()
                                              if charge != 0.0)
            # 统计量越大表示非零 Q 模板比零 Q 模板改进越明显。
            scores[q].append(statistic)
            all_fit_scores[q].append(fit.scores)
            fitted_charges[q].append(fit.q_e)
            records.append({"true_q_e": q, "repeat": repeat, "seed": case.run.seed,
                            "subset": "calibration" if repeat < calibration_count else "validation",
                            "statistic": statistic, "fitted_q_e": fit.q_e,
                            "null_score": fit.scores[0.0],
                            "best_nonzero_score": fit.scores[0.0] - statistic,
                            "all_scores_json": json.dumps(fit.scores)})
    null_calibration = np.asarray(scores[0.0][:calibration_count])
    # 先在零电荷校准半样本上选第 95 百分位阈值；再用留出半样本
    # 分别估计假阳性率和各备择的检出概率。
    threshold = float(np.quantile(null_calibration, 0.95, method="higher"))
    false_positive = float(np.mean(np.asarray(scores[0.0][calibration_count:]) > threshold))
    power = {q: float(np.mean(np.asarray(values[calibration_count:]) > threshold))
             for q, values in scores.items() if q != 0.0}
    minimum = (calibrated_detection_limit(
        null_calibration,
        {q: np.asarray(v[calibration_count:]) for q, v in scores.items() if q},
        false_positive=0.05, power=0.95)
        if false_positive <= 0.05 else None)
    ci_thresholds = {}
    # 对每个真实 Q 分别校准“该 Q 分数距最佳分数”阈值，构成网格区间。
    for q in templates:
        differences = [item[q] - min(item.values())
                       for item in all_fit_scores[q][:calibration_count]]
        ci_thresholds[q] = float(np.quantile(differences, 0.95, method="higher"))
    coverage = {}
    bias = {}
    for q in templates:
        held_out = all_fit_scores[q][calibration_count:]
        covered = []
        for item in held_out:
            minimum_score = min(item.values())
            accepted = [charge for charge in templates
                        if item[charge] - minimum_score <= ci_thresholds[charge]]
            # 离散网格可能不连续；此处报告被接受 Q 的凸包覆盖情况。
            covered.append(bool(accepted) and min(accepted) <= q <= max(accepted))
        coverage[q] = float(np.mean(covered))
        bias[q] = float(np.mean(np.asarray(fitted_charges[q][calibration_count:]) - q))
    summary = DetectionCalibration(threshold, false_positive, power, minimum,
                                   repetitions, calibration_count,
                                   repetitions - calibration_count, bias, coverage)
    with (output / "calibration_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "calibration.json").write_text(json.dumps({
        "threshold": threshold, "false_positive_rate": false_positive,
        "power_by_charge": power, "minimum_detectable_abs_q_e": minimum,
        "repetitions": repetitions,
        "calibration_repetitions": calibration_count,
        "validation_repetitions": repetitions - calibration_count,
        "charge_bias_by_true_q_e": bias,
        "interval_coverage_by_true_q_e": coverage,
        "interval_method": "Charge-grid likelihood-score-difference thresholds learned on calibration trials; reported interval is the hull of accepted grid points on held-out trials.",
        "interpretation": "Conditional on this charge grid, source, detector, template Monte Carlo, and static droplet.",
        "warning": "Fewer than 100 repetitions provide weak tail-probability calibration."
                   if repetitions < 100 else None,
    }, indent=2), encoding="utf-8")
    return summary
