"""检查已交付的 35 份配置；--print-patch 仅输出新增配置补丁，不启动输运。

矩阵定义集中在 study.planned_cases。生成器默认只比较文件，不覆盖人工编辑；
开发者可将新增文件补丁交给 apply_patch。正式计算直接读取完整 YAML。
"""

from pathlib import Path
import argparse
import json
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from droplet_shadow.study import manifest_data, planned_cases

COMMENTS = {
    "radius_um": "液滴半径 μm", "l1_mm": "束腰到液滴 mm", "l2_mm": "液滴到屏幕 mm",
    "temperature_k": "水温 K", "epsilon_water": "水的相对介电常数，连续介质假设",
    "surface_tension_n_m": "仅用于 Rayleigh 比例，N/m",
    "q_e": "整个液滴净电荷，以 e 为单位，不是双层单壳电荷",
    "double_layer_q_e": "内壳总电荷 e；外壳为相反总电荷", "double_layer_thickness_nm": "双壳径向间隔 nm",
    "dipole_potential_v": "球内相对球外的等效势差 V，不是瞬时分子局域场",
    "dipole_thickness_nm": "等效极化层厚度 nm，属于模型假设", "dipole_epsilon": "等效层相对介电常数，假设值",
    "ion_positive_count": "PB 球内正离子总数，非体浓度", "ion_negative_count": "PB 球内负离子总数",
    "surface_fixed_e": "PB 固定表面电荷 e，不属于移动离子",
    "energy_mev": "初级电子平均动能 MeV", "energy_spread_rms": "相对 RMS 能散，0.001=0.1%",
    "crossover_fwhm_um": "每个横向方向的焦点 FWHM，μm",
    "divergence_rms_mrad": "每个横向方向 RMS 发散角 mrad；固定此值缩焦意味着降低发射度",
    "x_xprime_correlation": "x 与斜率 x' 的相关系数；束腰取 0", "y_yprime_correlation": "y 与斜率 y' 的相关系数",
    "psf_fwhm_um": "0=显式关闭 PSF；仍保留有限源、散射和像素积分",
    "pixel_um": "探测器平面的像素宽度 μm", "diameter_mm": "圆形有效探测器直径 mm",
    "efficiency": "探测效率，未实测情景值", "gain_mean": "平均增益，任意信号单位",
    "gain_shape": "单电子 Gamma 增益的 shape 参数", "background_counts_pixel": "每像素平均背景，假设值",
    "read_noise_rms": "每像素读出噪声 RMS，假设值",
    "engine": "正式计算使用 Geant4；配置文件本身不会启动计算",
    "n_simulated": "实际追踪 1000 万个初级事件，包括无落点事件",
    "exposure_electrons": "主展示曝光；另由相同落点生成 10^5 和 10^7 曝光",
    "seed": "源/输运种子，版本及线程数需共同记录", "raster_width_mm": "保存视野，覆盖有效探测器",
    "geant4_executable": "null 使用项目 build/droplet_g4；在 Linux 重新编译",
    "geant4_threads": "0 自动留一核、最多 8；条件间不并行",
    "show_progress": "显示事件完成数和处理阶段", "droplet_material": "water 保留材料输运；vacuum 移除材料",
    "transport_step_um": "仅 Python ray 使用，本研究不改变 Geant4 精度",
    "geant4_world_step_mm": "远场最大步长 mm", "geant4_far_step_mm": "中远场最大步长 mm",
    "geant4_near_step_um": "近场最大步长 μm", "geant4_core_step_um": "液滴核心最大步长 μm",
    "geant4_cut_um": "次级生产 range cut，μm；不是每个电子的硬性停止距离",
    "geant4_layer_step_scale": "界面限步倍率；不能为加速而放宽",
    "model": "surface / volume / neutral_double_layer / poisson_boltzmann",
}


def documents():
    cases = planned_cases()
    folder = Path("examples/field_scale_study")
    outputs = {folder / "manifest.json": json.dumps(manifest_data(cases), ensure_ascii=False, indent=2) + "\n"}
    for case in cases:
        lines = [f"# {case.case_id}", f"# {case.note}",
                 "# 状态：未运行。参考出处：data/field_scale_sources.json；运行说明：docs/linux_field_study.md。"]
        for line in yaml.safe_dump(case.config.to_dict(), sort_keys=False, allow_unicode=True).splitlines():
            key = line.strip().split(":", 1)[0]
            lines.append(line + ("  # " + COMMENTS[key] if key in COMMENTS else ""))
        outputs[folder / (case.case_id + ".yaml")] = "\n".join(lines) + "\n"
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-patch", action="store_true")
    args = parser.parse_args()
    outputs = documents()
    if args.print_patch:
        print("*** Begin Patch")
        for relative, content in outputs.items():
            if (ROOT / relative).exists():
                raise FileExistsError(f"Refusing to replace {relative}")
            print(f"*** Add File: {relative}")
            print("\n".join("+" + line for line in content.splitlines()))
        print("*** End Patch")
    else:
        changed = [str(p) for p, text in outputs.items() if not (ROOT / p).is_file() or (ROOT / p).read_text(encoding="utf-8") != text]
        if changed:
            raise SystemExit("Missing/changed generated documents (not overwritten):\n" + "\n".join(changed))
        print("35 YAML files and manifest match the planned matrix; no transport invoked.")


if __name__ == "__main__":
    main()
