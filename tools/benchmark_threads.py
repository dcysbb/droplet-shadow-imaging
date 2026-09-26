"""复现 Geant4 线程性能或统计一致性；绝不通过放宽步长来获得加速。

性能：python tools/benchmark_threads.py --n 10000 --repetitions 3
统计：python tools/benchmark_threads.py --statistics --n 10000 --repetitions 5
输出包含每次完整结果、基准配置、线程数、分阶段时间及汇总 JSON。
性能的首次运行也保留；多次中位数降低系统缓存、温度和调度的影响。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from droplet_shadow import load_config, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/baseline-geant4.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/thread-benchmark"))
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--statistics", action="store_true")
    args = parser.parse_args()
    if args.n < 1 or args.repetitions < (5 if args.statistics else 1):
        parser.error("Positive n required; statistics requires at least 5 repetitions")
    # 禁止把不同测试结果不小心覆盖；需要复跑时选新的 --output。
    if args.output.exists():
        parser.error("Output already exists; choose a new --output directory")
    args.output.mkdir(parents=True)
    base = load_config(args.config).with_updates(run={"engine": "geant4", "n_simulated": args.n,
        "exposure_electrons": args.n, "raster_width_mm": 40})
    # 对照均从同一几何/源出发；删除 PB 状态，避免残留模型参数。
    cleared = dict(surface_fixed_e=0, ion_positive_count=0, ion_negative_count=0,
                   dipole_potential_v=0)
    cases = {
        "surface": base.with_updates(charge={**cleared, "model": "surface", "q_e": 1e6,
                                             "double_layer_q_e": 0}),
        "double_layer_10nm": base.with_updates(charge={**cleared, "model": "neutral_double_layer",
            "q_e": 0, "double_layer_q_e": 1e8, "double_layer_thickness_nm": 10}),
    }
    threads = [1, 4] if args.statistics else [1, 2, 4, 0]
    rows = []
    for name, config in cases.items():
        for repeat in range(args.repetitions):
            # 交替顺序，减小升温、缓存及后台负载与线程数的系统关联。
            for requested in threads if repeat % 2 == 0 else list(reversed(threads)):
                cfg = config.with_updates(run={"geant4_threads": requested,
                                               "seed": config.run.seed + repeat * 101})
                path = args.output / f"{name}_t{requested}_r{repeat}"
                start = time.perf_counter()
                result = simulate(cfg, path)
                native = result.runtime["geant4"]
                row = {"case": name, "repeat": repeat, "seed": cfg.run.seed,
                       "requested_threads": requested, "actual_threads": native["actual_threads"],
                       "end_to_end_s": time.perf_counter() - start,
                       "transport_s": native["transport_s"], "initialization_s": native["initialization_s"],
                       "input_preparation_s": native["input_preparation_s"],
                       "merge_read_s": native["merge_read_s"],
                       "detector_s": result.runtime["timings_s"]["detector"],
                       "save_s": result.runtime["timings_s"]["save"],
                       "metrics": result.metrics}
                if args.statistics:
                    # 使用无噪声期望图，避免另一次曝光抽样混入输运对照。
                    image = result.images["expected"]
                    edges = result.images["x_edges_m"]
                    centers = (edges[1:] + edges[:-1]) / 2
                    r = np.hypot(centers[:, None], centers[None, :])
                    profile = np.histogram(r, bins=np.linspace(0, .02, 9), weights=image)[0] / args.n
                    row["radial_signal_per_primary"] = profile.tolist()
                rows.append(row)
                (args.output / "runs.json").write_text(json.dumps(rows, indent=2))
                print(f"{name} repeat={repeat} threads={requested}->{native['actual_threads']} "
                      f"transport={row['transport_s']:.3f}s total={row['end_to_end_s']:.3f}s", flush=True)
                del result
    summary = {"mode": "statistics" if args.statistics else "performance", "n": args.n,
               "repetitions": args.repetitions, "cases": {}}
    metric_names = ["primary_detector_fraction", "secondary_detector_fraction", "primary_exit_fraction",
                    "mean_primary_exit_loss_kev", "primary_exit_scattering_rms_mrad"]
    for name in cases:
        selected = [row for row in rows if row["case"] == name]
        if args.statistics:
            # 不同 seed 是独立重复；同一 seed 的串/并行做配对差，
            # 用重复间差异估计 MC 不确定度。4 SE 仅是回归筛查，不是精度认证。
            paired = {t: sorted([r for r in selected if r["requested_threads"] == t],
                                key=lambda r: r["repeat"]) for t in threads}
            checks = {}
            for key in metric_names + [f"radial_{i}" for i in range(8)]:
                def get(row):
                    return (row["radial_signal_per_primary"][int(key[7:])] if key.startswith("radial_")
                            else row["metrics"].get(key))
                a, b = [list(map(get, paired[t])) for t in threads]
                if None in a or None in b:
                    checks[key] = {"status": "undefined; increase sampling"}
                    continue
                delta = np.asarray(b) - np.asarray(a)
                mean, se = float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))
                checks[key] = {"difference": mean, "standard_error": se,
                               "within_4se": abs(mean) <= 4 * se + 1e-12}
            summary["cases"][name] = checks
        else:
            baseline = [r for r in selected if r["requested_threads"] == 1]
            t1 = statistics.median(r["transport_s"] for r in baseline)
            total1 = statistics.median(r["end_to_end_s"] for r in baseline)
            summary["cases"][name] = [{
                "requested_threads": t, "actual_threads": group[0]["actual_threads"],
                "median_transport_s": statistics.median(r["transport_s"] for r in group),
                "median_end_to_end_s": statistics.median(r["end_to_end_s"] for r in group),
                "transport_speedup": t1 / statistics.median(r["transport_s"] for r in group),
                "end_to_end_speedup": total1 / statistics.median(r["end_to_end_s"] for r in group),
                "primaries_per_transport_s": args.n / statistics.median(r["transport_s"] for r in group),
            } for t in threads for group in [[r for r in selected if r["requested_threads"] == t]]]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
