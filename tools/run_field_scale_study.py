"""研究入口：默认列清单；--run 才输运，--analyze 只读取已完成结果。

此脚本不会因为打开 Notebook、导入模块或遗漏参数而启动 3.5 亿事件。
大规模运行必须同时指定 --run 和 --all 或 --case。Linux 先执行独立
validate_field_study.py --run 验证数值精度，再正式运行。
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from droplet_shadow.study import DEFAULT_MANIFEST, load_study, resource_estimate, run_case, study_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicitly start Geant4 transport")
    mode.add_argument("--analyze", action="store_true", help="Only postprocess existing complete results")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--all", action="store_true")
    select.add_argument("--case", action="append", help="Case ID; repeat for multiple cases")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "results/field_scale_study")
    args = parser.parse_args(argv)
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if args.run and not (args.all or args.case):
        parser.error("Choose --all or --case explicitly before running")
    cases = load_study(args.manifest)
    selected = set(args.case) if args.case else {c.case_id for c in cases}
    if unknown := selected - {c.case_id for c in cases}:
        parser.error(f"Unknown case IDs: {sorted(unknown)}")
    chosen = [c for c in cases if c.case_id in selected]
    if args.analyze:
        from droplet_shadow.study_analysis import analyze_study
        # 自动补入只读参照；缺少参照结果时跳过比较，绝不补跑参照。
        selected |= {ref for c in chosen for ref in (c.neutral_case, c.absent_case)}
        for name, status in analyze_study(cases, args.output, selected=selected).items():
            print(name, status)
        return
    for c in chosen:
        estimate = resource_estimate(c.config)
        print(f"{c.case_id:30} {c.group:15} N={c.config.run.n_simulated:,} "
              f"PSF={c.config.detector.psf_fwhm_um:g} μm "
              f"estimated RAM={estimate['estimated_peak_memory_bytes']/1e9:.1f} GB")
    disk = sum(resource_estimate(c.config)["estimated_disk_bytes"] for c in chosen)
    print(f"Selected {len(chosen)} cases; {sum(c.config.run.n_simulated for c in chosen):,} primaries; "
          f"estimated disk {disk/1e9:.1f} GB (not a guarantee).")
    if not args.run:
        print("CHECK ONLY — no transport started; use --run --case ID or --run --all on Linux.")
        return
    with study_lock(args.output):
        for index, case in enumerate(chosen, 1):
            print(f"[{index}/{len(chosen)}] {case.case_id}", flush=True)
            run_case(case, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
