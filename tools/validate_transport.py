"""检查 Geant4 水中散射分布及数值参数的敏感性。

四种运行使用相同 3 MeV、中性、中心铅笔束：基准、缩短水内
最大步长、收紧和放宽次级粒子生产截断。均在水滴出口统计
初级电子，减少屏幕孔径造成的选择偏差。KS 检验只做分布
差异诊断；有限样本下 p 值大不证明两个设置完全等价。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from droplet_shadow.config import SimulationConfig
from droplet_shadow.constants import C, E_CHARGE, M_E
from droplet_shadow.native import run_geant4


def main() -> None:
    """重复输运、汇总能损/角分布、输出 JSON 与散射角图。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "transport_validation")
    args = parser.parse_args()
    kinetic = 3e6 * E_CHARGE
    p = np.sqrt(kinetic * (kinetic + 2 * M_E * C**2)) / C
    phase = np.zeros((args.n, 4))
    momentum = np.tile([0.0, 0.0, p], (args.n, 1))
    baseline = SimulationConfig().with_updates(charge={"q_e": 0},
        run={"engine": "geant4", "n_simulated": args.n, "exposure_electrons": args.n})
    cases = {
        # 除括号中的单一参数外，各组配置完全相同，便于归因。
        "baseline": baseline,
        "half_water_step": baseline.with_updates(run={"geant4_core_step_um": 0.25}),
        "tenth_production_cut": baseline.with_updates(run={"geant4_cut_um": 0.1}),
        "coarse_production_cut": baseline.with_updates(run={"geant4_cut_um": 100.0}),
    }
    report = {"n_primary_per_case": args.n, "cases": {}}
    distributions = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, config in cases.items():
        hits = run_geant4(config, phase, momentum, args.output / name / "native")
        primary = hits["exit_parent_id"] == 0
        # 角度相对初始 +z 方向；出口统计不依赖是否打中屏幕。
        loss_kev = (3.0 - hits["exit_energy_mev"][primary]) * 1000
        angle_mrad = np.arccos(np.clip(hits["exit_uz"][primary], -1, 1)) * 1000
        angle_rms = float(np.sqrt(np.mean(angle_mrad**2)))
        distributions[name] = angle_mrad
        report["cases"][name] = {
            "water_step_um": config.run.geant4_core_step_um,
            "production_cut_um": config.run.geant4_cut_um,
            "primary_exits": int(primary.sum()),
            "mean_energy_loss_keV": float(np.mean(loss_kev)),
            "energy_loss_standard_error_keV": float(np.std(loss_kev, ddof=1) / np.sqrt(len(loss_kev))),
            "scattering_median_mrad": float(np.median(angle_mrad)),
            "scattering_rms_mrad": angle_rms,
            "scattering_rms_standard_error_mrad": float(
                np.std(angle_mrad**2, ddof=1) / (2 * angle_rms * np.sqrt(len(angle_mrad)))),
            "scattering_p90_mrad": float(np.quantile(angle_mrad, 0.9)),
        }
    for name in ("half_water_step", "tenth_production_cut", "coarse_production_cut"):
        # 两样本 KS 只比较散射角的经验分布形状。
        report["cases"][name]["scattering_ks_p_vs_baseline"] = float(
            ks_2samp(distributions["baseline"], distributions[name]).pvalue)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, angles in distributions.items():
        clipped = angles[angles < 300]
        ax.hist(clipped, bins=np.linspace(0, 300, 100), density=True,
                histtype="step", linewidth=1.4, label=name.replace("_", " "))
    ax.set(xlabel="Primary exit scattering angle (mrad)",
           ylabel="Density (1/mrad)", title="3 MeV electrons through 100 μm liquid water")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "scattering_convergence.png", dpi=160)
    fig.savefig(args.output / "scattering_convergence.svg")
    plt.close(fig)
    (args.output / "transport_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
