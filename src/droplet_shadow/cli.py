"""面向实验使用者的命令行入口。

``simulate`` 运行单点，``scan`` 运行可缓存的参数扫描，
``analyze`` 比较同条件图像，``fit-charge`` 做离散电荷模板拟合，
``estimate-limit`` 用独立曝光校准检出限。核心算法均在同名 Python
模块中，CLI 只负责解析参数、调用与打印，不应另写一套物理公式。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import numpy as np

from .analysis import fit_charge, image_metrics
from .config import load_config
from .simulation import load_images, simulate, simulate_controls, version_fingerprint
from .sensitivity import detection_calibration


SCAN_VALUES = {
    # 首轮预设网格；完整五维笛卡尔积可用 --full，但计算量很大。
    "energy_mev": [1, 2, 3, 5, 10],
    "q_e": [-1e7, -1e6, -1e5, -1e4, 0, 1e4, 1e5, 1e6, 1e7],
    "l1_mm": [5, 10, 20, 30],
    "l2_mm": [300, 500, 800],
    "crossover_fwhm_um": [5, 10, 20],
}
SECTION_FOR = {"energy_mev": "source", "q_e": "charge", "l1_mm": "geometry",
               "l2_mm": "geometry", "crossover_fwhm_um": "source"}


def scan(config_path: str | Path, output_dir: str | Path,
         n_simulated: int | None = None, full: bool = False,
         q_values: list[float] | None = None) -> list[dict[str, object]]:
    """按阶段执行能量–电荷网格及单因素扫描，保存逐点指标 CSV。

若结果文件存在、配置完全相同且源代码/Geant4 哈希未变，复用缓存；
否则重新计算。缓存只保证计算版本一致，不代替 Monte Carlo 误差估计。
"""
    config = load_config(config_path)
    if n_simulated is not None:
        config = config.with_updates(run={"n_simulated": n_simulated})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = version_fingerprint(config)
    values = {key: list(items) for key, items in SCAN_VALUES.items()}
    if q_values is not None:
        values["q_e"] = sorted(set(q_values))
    if config.charge.model == "poisson_boltzmann" and any(
        q != config.charge.q_e for q in values["q_e"]
    ):
        raise ValueError("PB charge scans require new ion counts at each Q; use separate configs")
    cases = []
    if full:
        # 最昂贵模式：5 个参数所有组合；默认只做二维主扫描和单因素。
        cases = [("full", settings) for settings in product(
            values["energy_mev"], values["q_e"], values["l1_mm"],
            values["l2_mm"], values["crossover_fwhm_um"])]
    else:
        for parameter in ("l1_mm", "l2_mm", "crossover_fwhm_um"):
            for value in values[parameter]:
                cases.append((parameter, value))
        # The energy-charge grid is the primary two-dimensional pass.
        cases = [("energy_q", (energy, q)) for energy in values["energy_mev"]
                 for q in values["q_e"]] + cases
    results = []
    for number, (parameter, value) in enumerate(cases):
        if parameter == "full":
            energy, q, l1, l2, width = value
            adjusted = config.with_updates(source={"energy_mev": energy,
                                                   "crossover_fwhm_um": width},
                                           charge={"q_e": q},
                                           geometry={"l1_mm": l1, "l2_mm": l2})
            label = (f"E{energy:g}_Q{q:+.0f}_L1{l1:g}_L2{l2:g}_S{width:g}")
        elif parameter == "energy_q":
            energy, q = value
            adjusted = config.with_updates(source={"energy_mev": energy}, charge={"q_e": q})
            label = f"energy_{energy:g}_charge_{q:+.0f}"
        else:
            adjusted = config.with_updates(**{SECTION_FOR[parameter]: {parameter: value}})
            label = f"{parameter}_{value:+g}"
        case_dir = output / label
        cached = None
        if ((case_dir / "result.json").exists() and
                all((case_dir / name).exists() for name in ("hits.npz", "images.npz", "shadow.png"))):
            candidate = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
            if (candidate.get("config") == adjusted.to_dict() and
                    all(candidate.get("versions", {}).get(key) == value
                        for key, value in fingerprint.items())):
                cached = candidate
        if cached is None:
            result = simulate(adjusted, case_dir)
            row = {"case": label, "parameter": parameter, "value": str(value), **result.metrics}
        else:
            row = {"case": label, "parameter": parameter, "value": str(value), **cached["metrics"]}
        results.append(row)
        fields = sorted(set().union(*(item.keys() for item in results)))
        with (output / "scan_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"[{number+1}/{len(cases)}] {label}", flush=True)
    return results


def main() -> None:
    """定义子命令并把参数映射到 Python API。"""
    parser = argparse.ArgumentParser(prog="droplet-shadow")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="Run one exposure")
    sim.add_argument("--config", required=True)
    sim.add_argument("--output", required=True)
    sim.add_argument("--controls", action="store_true", help="Also run neutral and no-droplet controls")
    scan_parser = sub.add_parser("scan", help="Run resumable staged parameter scans")
    scan_parser.add_argument("--config", required=True)
    scan_parser.add_argument("--output", required=True)
    scan_parser.add_argument("--n-simulated", type=int)
    scan_parser.add_argument("--full", action="store_true", help="All five-dimensional combinations")
    scan_parser.add_argument("--q-values", type=float, nargs="+",
                             help="Custom charge grid in elementary charges")
    ana = sub.add_parser("analyze", help="Compare image with a reference")
    ana.add_argument("--result", required=True)
    ana.add_argument("--reference", required=True)
    fit = sub.add_parser("fit-charge", help="Fit Q against same-condition image templates")
    fit.add_argument("--observed", required=True)
    fit.add_argument("--template", action="append", nargs=2, metavar=("Q_E", "DIRECTORY"), required=True)
    limit = sub.add_parser("estimate-limit", help="Independent-exposure 5%%/95%% charge sensitivity")
    limit.add_argument("--config", required=True)
    limit.add_argument("--template", action="append", nargs=2,
                       metavar=("Q_E", "DIRECTORY"), required=True)
    limit.add_argument("--repetitions", type=int, default=100)
    limit.add_argument("--output", required=True)
    limit.add_argument("--no-nuisance", action="store_true")
    args = parser.parse_args()
    if args.command == "simulate":
        config = load_config(args.config)
        if args.controls:
            controls = simulate_controls(config, args.output)
            print(json.dumps({name: result.metrics for name, result in controls.items()},
                             ensure_ascii=False, indent=2))
        else:
            result = simulate(config, args.output)
            print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    elif args.command == "scan":
        scan(args.config, args.output, args.n_simulated,
             full=args.full, q_values=args.q_values)
    elif args.command == "analyze":
        images = load_images(args.result)
        ref = load_images(args.reference)
        config = load_config(args.result + "/config.yaml") if (Path(args.result) / "config.yaml").exists() else None
        if config is None:
            # 旧结果目录可能没有独立 config.yaml；以 result.json 为准重建。
            from .config import SimulationConfig, Geometry, Charge, Source, Detector, Run
            info = json.loads((Path(args.result) / "result.json").read_text(encoding="utf-8"))
            sections = info["config"]
            config = SimulationConfig(Geometry(**sections["geometry"]), Charge(**sections["charge"]),
                                      Source(**sections["source"]), Detector(**sections["detector"]),
                                      Run(**sections["run"]))
        print(json.dumps(image_metrics(config, images["expected"], ref["expected"],
                                       images["x_edges_m"]), ensure_ascii=False, indent=2))
    elif args.command == "fit-charge":
        observed = load_images(args.observed)["observed"]
        templates = {float(q): load_images(path)["expected"] for q, path in args.template}
        result = fit_charge(observed, templates)
        print(json.dumps({"q_e": result.q_e, "score": result.score,
                          "scores": result.scores}, ensure_ascii=False, indent=2))
    else:
        result = detection_calibration(
            load_config(args.config), {float(q): path for q, path in args.template},
            args.output, repetitions=args.repetitions,
            fit_nuisance=not args.no_nuisance)
        print(json.dumps({"threshold": result.threshold,
                          "false_positive_rate": result.false_positive_rate,
                          "power_by_charge": result.power_by_charge,
                          "minimum_detectable_abs_q_e": result.minimum_detectable_abs_q_e,
                          "charge_bias_by_true_q_e": result.charge_bias_by_true_q_e,
                          "interval_coverage_by_true_q_e": result.interval_coverage_by_true_q_e,
                          "calibration_repetitions": result.calibration_repetitions,
                          "validation_repetitions": result.validation_repetitions,
                          "warning": "Fewer than 100 repeats: this is only a software smoke test"
                                     if result.repetitions < 100 else None},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
