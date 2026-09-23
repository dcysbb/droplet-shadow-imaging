"""用 NIST ESTAR 液态水数据核对 Geant4 的 3 MeV 能损量级。

这里使用沿球直径穿过 100 μm 水的铅笔束；统计初级电子在
**离开水滴时**的能量，不按探测器是否接收筛选。ESTAR 的
18.89 keV 是连续减速近似参考值，不能要求逐粒子完全相同。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from droplet_shadow.config import SimulationConfig
from droplet_shadow.constants import C, E_CHARGE, M_E
from droplet_shadow.native import run_geant4


def main() -> None:
    """运行中性水滴铅笔束，保存 NIST/Geant4 对照与出口散射指标。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stopping_validation")
    args = parser.parse_args()
    config = SimulationConfig().with_updates(
        charge={"q_e": 0}, source={"energy_mev": 3.0},
        run={"engine": "geant4", "n_simulated": args.n,
             "exposure_electrons": args.n})
    kinetic = 3e6 * E_CHARGE
    # 相对论动量由动能得到；所有电子从束轴中心直射，消除源展宽。
    momentum = np.sqrt(kinetic * (kinetic + 2 * M_E * C**2)) / C
    beam = np.tile([0.0, 0.0, momentum], (args.n, 1))
    hits = run_geant4(config, np.zeros((args.n, 4)), beam, args.output / "native")
    exiting_primary = hits["exit_energy_mev"][hits["exit_parent_id"] == 0]
    # 只比较初级电子；次级粒子有自己的初始能量与路径长度。
    primary_mask = hits["exit_parent_id"] == 0
    angles = np.arccos(np.clip(hits["exit_uz"][primary_mask], -1, 1))
    mean_loss_kev = float(np.mean(3.0 - exiting_primary) * 1000)
    # NIST ESTAR: water (liquid), 3 MeV, total mass stopping power
    # 1.889 MeV cm²/g; 100 μm water at 1 g/cm³ gives 18.89 keV.
    reference_kev = 1.889 * 0.01 * 1000
    report = {
        "source": "https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html",
        "estar_material": "Water, Liquid (material 276)",
        "energy_MeV": 3.0,
        "estar_collision_MeV_cm2_g": 1.846,
        "estar_radiative_MeV_cm2_g": 0.04299,
        "estar_total_MeV_cm2_g": 1.889,
        "path_um": 100.0,
        "reference_mean_loss_keV": reference_kev,
        "geant4_mean_primary_exit_loss_keV": mean_loss_kev,
        "ratio_geant4_to_estar": mean_loss_kev / reference_kev,
        "n_primary": args.n,
        "n_primary_exited": len(exiting_primary),
        "n_primary_at_detector": int(hits["primary"].sum()),
        "exit_scattering_rms_mrad": float(np.sqrt(np.mean(angles**2)) * 1000),
        "exit_scattering_median_mrad": float(np.median(angles) * 1000),
        "scope": "ESTAR continuous-slowing-down mean versus finite-slab Monte Carlo exit energy; secondary production cut and straggling can shift the comparison.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stopping_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
