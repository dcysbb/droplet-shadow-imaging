"""只求静态场并导出独立 PNG/SVG，不调用任何电子输运。

14 个不同模型各绘一幅图；焦点和探测距不改变液滴静态场，因此无需
把同一电场重复画 35 次。这些图不应解释为 shadow image 或输运验收。
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from droplet_shadow.study import load_study
from droplet_shadow.study_analysis import field_figure, save_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/field_scale_static")
    args = parser.parse_args()
    models = {}
    for case in load_study():
        models.setdefault(case.model, case)
    for model, case in models.items():
        figure = field_figure(case)
        save_figure(figure, args.output / model)
        plt.close(figure)
    print(f"Saved {len(models)} static-model PNG/SVG pairs; no electron transport.")


if __name__ == "__main__":
    main()
