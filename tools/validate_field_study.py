"""Linux 上显式运行的无材料确定性探针：默认只说明，不启动 Geant4。

只检查静电积分的数值收敛，不判断图像可见性。每个事件从理想点源
出发，b 是无场轨迹在液滴中心平面的几何半径；实际最近距离会被场改变。
输出普通/收紧步长落点、位移和能量误差，供人工复核。
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_probes(case, destination):
    from droplet_shadow.constants import C, E_CHARGE, M_E
    from droplet_shadow.fields import build_field
    from droplet_shadow.native import run_geant4
    from droplet_shadow.study import atomic_json
    from droplet_shadow.study_analysis import save_figure
    import matplotlib.pyplot as plt

    cfg = case.config
    R = cfg.geometry.radius_m
    # 两侧对数加密；最小 0.1 nm，不使用浮点 nextafter 充当可物理解读的距离。
    offsets = np.geomspace(.1e-9, 10e-6, 150)
    b = np.unique(np.r_[np.linspace(0, 2 * R, 180), R - offsets, R + offsets])
    slopes = b / (cfg.geometry.l1_mm * 1e-3)
    phase = np.zeros((len(b), 4))
    phase[:, 2] = slopes
    kinetic = cfg.source.energy_mev * 1e6 * E_CHARGE
    p = np.sqrt(kinetic * (kinetic + 2 * M_E * C**2)) / C
    direction = np.column_stack((slopes, np.zeros_like(b), np.ones_like(b)))
    momentum = p * direction / np.linalg.norm(direction, axis=1)[:, None]
    base = cfg.with_updates(run={"droplet_material": "vacuum", "n_simulated": len(b),
                                 "exposure_electrons": len(b)})
    smaller = base.with_updates(run={name: getattr(base.run, name) / 2 for name in (
        "geant4_world_step_mm", "geant4_far_step_mm", "geant4_near_step_um",
        "geant4_core_step_um", "geant4_layer_step_scale")})
    destination.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    normal = run_geant4(base, phase, momentum, destination / "normal")
    refined = run_geant4(smaller, phase, momentum, destination / "refined")
    for hits in (normal, refined):
        if not np.array_equal(hits["event_id"], np.arange(len(b))) or not hits["primary"].all():
            raise RuntimeError("A deterministic vacuum probe is missing or duplicated")
    shift = refined["x_m"] - b * cfg.geometry.magnification
    delta = abs(refined["x_m"] - normal["x_m"])
    field = build_field(cfg)
    phi0 = field.potential(np.array([[0., 0., -cfg.geometry.l1_mm * 1e-3]]))[0]
    final_xyz = np.column_stack((refined["x_m"], refined["y_m"], np.full(len(b), cfg.geometry.l2_mm * 1e-3)))
    energy_error_ev = (refined["energy_mev"] - cfg.source.energy_mev) * 1e6 - (field.potential(final_xyz) - phi0)
    np.savez_compressed(destination / "probes.npz", b_m=b, normal_x_m=normal["x_m"],
                        refined_x_m=refined["x_m"], displacement_m=shift,
                        landing_difference_m=delta, energy_error_ev=energy_error_ev)
    # 相对误差只在非零信号上定义；弱信号仍受绝对误差条件约束。
    significant = abs(shift) > 1e-6
    report = {"case_id": case.case_id, "formal_mc_run": False, "n_probes": len(b),
              "elapsed_s": time.perf_counter() - start, "config": base.to_dict(),
              "refined_config": smaller.to_dict(),
              "max_landing_difference_um": float(delta.max() * 1e6),
              "max_energy_error_ev": float(abs(energy_error_ev).max()),
              "max_relative_landing_change_for_shift_gt_1um": float(np.max(delta[significant] / abs(shift[significant]))) if significant.any() else None,
              "landing_below_0p01_pixel": bool(np.all(delta < .01 * cfg.detector.pixel_um * 1e-6)),
              "energy_below_0p02_ev": bool(np.all(abs(energy_error_ev) < .02)),
              "relative_below_1percent": bool(np.all(delta[significant] < .01 * abs(shift[significant])))}
    atomic_json(destination / "validation.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(b * 1e6, shift * 1e6, color="#214e7a")
    axes[0].set(xlabel="Unperturbed impact b (μm)", ylabel="Detector displacement (μm)")
    axes[1].plot((b - R) * 1e9, shift * 1e6, color="#214e7a")
    axes[1].set(xlim=(-30, 30), xlabel="b − R (nm)", ylabel="Detector displacement (μm)")
    fig.suptitle(case.case_id + " — vacuum probes, not water shadow image")
    save_figure(fig, destination / "probe_displacement")
    plt.close(fig)
    return report


def main(argv=None):
    from droplet_shadow.study import DEFAULT_MANIFEST, load_study, atomic_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly start vacuum Geant4 probes")
    parser.add_argument("--case", action="append")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "results/field_scale_validation")
    args = parser.parse_args(argv)
    cases = load_study(args.manifest)
    chosen = [c for c in cases if c.case_id in args.case] if args.case else [c for c in cases if c.config.source.crossover_fwhm_um == 10 and c.config.geometry.l2_mm == 2000]
    if args.case and set(args.case) - {c.case_id for c in chosen}:
        parser.error("Unknown case ID")
    print("Vacuum probe cases:", ", ".join(c.case_id for c in chosen))
    if not args.run:
        print("CHECK ONLY — add --run on Linux to execute probes; no transport invoked.")
        return
    if args.output.exists():
        parser.error("Output exists; choose a new directory to preserve earlier diagnostics")
    args.output.mkdir(parents=True)
    reports = [run_probes(c, args.output / c.case_id) for c in chosen]
    atomic_json(args.output / "summary.json", {"reports": reports})
    if not all(r["landing_below_0p01_pixel"] and r["energy_below_0p02_ev"] and r["relative_below_1percent"] for r in reports):
        raise SystemExit("Numerical checks failed; inspect diagnostics before formal transport")
    print("Vacuum probe checks passed. This does not validate water scattering or signal detectability.")


if __name__ == "__main__":
    main()
