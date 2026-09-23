"""四个教学 Notebook 的源模板。

每个 ``notebook(name, cells)`` 调用由 Markdown 解释单元和 Python
代码单元组成。修改这里以后需重新运行本脚本生成 .ipynb，并运行
``execute_notebooks.py`` 保存结果；生成步骤会覆盖现有 Notebook，
所以不要只在 .ipynb 中手改、却忘记更新这个模板。
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks"
OUTPUT.mkdir(exist_ok=True)


def notebook(name: str, cells: list[tuple[str, str]]) -> None:
    """将 ('markdown'|'code', 内容) 序列写成可校验的 Jupyter 文件。"""
    item = nbf.v4.new_notebook()
    item.cells = [nbf.v4.new_markdown_cell(body) if kind == "markdown"
                  else nbf.v4.new_code_cell(body) for kind, body in cells]
    item.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    item.metadata.language_info = {"name": "python"}
    nbf.validate(item)
    nbf.write(item, OUTPUT / name)


# 每本都插入相同的初始化单元：定位项目根、导入源码、统一绘图风格。
# 允许从项目根目录或 notebooks/ 目录启动 Jupyter。
setup = """# 所有 Notebook 都从项目根目录读取配置和写入 results/。
from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
# 不要求先 pip install；优先直接导入当前工作树中的源码。
sys.path.insert(0, str(ROOT / 'src'))
import droplet_shadow
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'figure.figsize': (7, 4), 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})"""

notebook("01_literature_fields.ipynb", [
    ("markdown", "# 文献约束的单液滴电场模型\n\n## tl;dr\n下面直接计算同净电荷球壳/体电荷、严格中性双电层与有限厚度偶极层的径向电场。文献数值与适用环境见 `../data/literature_parameters.csv`。"),
    ("markdown", "## Context & Methods\n\n以半径 50 μm 的水滴为基准；局域探针场与远程 Coulomb 场分别参数化。参考 [JCP 2020](https://doi.org/10.1063/5.0006550)、[JPCL 2020](https://doi.org/10.1021/acs.jpclett.0c02061)、[Nature Communications 2022](https://doi.org/10.1038/s41467-021-27941-x) 和 [2024](https://doi.org/10.1038/s41467-024-47879-0)。"),
    ("code", setup),
    ("code", "# 创建基准几何，只把净电荷改为 10^7 e；M 是几何放大率。\nfrom droplet_shadow import SimulationConfig, build_field\nfrom droplet_shadow.fields import rayleigh_charge_e\nfrom droplet_shadow.config import Charge\nbase = SimulationConfig().with_updates(charge={'q_e': 1e7})\nprint(f'M={base.geometry.magnification:.1f}, Q/Q_R={1e7/rayleigh_charge_e(base.geometry.radius_m, 0.072):.3f}')"),
    ("markdown", "## Results\n\n### 同净电荷分布的内外场"),
    ("code", "# 只沿 x 轴取样；球对称模型在任何方位的径向场都相同。\nr_um = np.geomspace(0.1, 500, 500)\npoints = np.column_stack((r_um * 1e-6, np.zeros_like(r_um), np.zeros_like(r_um)))\n# 固定净 Q 的表面/体电荷，与零净 Q 的界面层形成对照。\nmodels = {'surface': base, 'volume': base.with_updates(charge={'model': 'volume'}),\n          'neutral double layer': base.with_updates(charge={'model': 'neutral_double_layer', 'q_e': 0, 'double_layer_q_e': 1e8}),\n          '1 V dipole layer': base.with_updates(charge={'model': 'surface', 'q_e': 0, 'dipole_potential_v': 1.0})}\nfig, ax = plt.subplots()\nfor name, cfg in models.items():\n    f = build_field(cfg)\n    ax.plot(r_um, abs(f.electric_field(points)[:, 0]) / 1e6, label=name)\nax.axvline(50, color='0.35', linestyle='--', label='droplet boundary')\nax.set(xscale='log', yscale='symlog', xlabel='Radius (μm)', ylabel='|E| (MV/m)', title='Radial field by charge model')\nax.legend(); fig.tight_layout(); plt.show()"),
    ("markdown", "### 守恒离子数的球对称 Poisson–Boltzmann 解"),
    ("code", "# -10^6 e 固定表面电荷与 +10^6 个移动反离子相抵：总净 Q=0。\nfrom droplet_shadow.fields import poisson_boltzmann\nneutral = base.with_updates(charge={'model':'poisson_boltzmann', 'q_e':0, 'surface_fixed_e':-1e6, 'ion_positive_count':1e6})\npb = poisson_boltzmann(neutral)\n# 100 μm > R=50 μm；严格球对称且净电荷为零时，外场应为零。\nprint('External E for neutral PB (V/m):', pb.electric_field(np.array([[100e-6,0,0]]))[0,0])\nfig, ax = plt.subplots()\nax.plot(pb.radii_m * 1e6, pb.field_v_m / 1e6, color='#214e7a')\nax.set(xlabel='Radius (μm)', ylabel='Internal radial E (MV/m)', title='Neutral fixed surface charge with mobile counterions')\nfig.tight_layout(); plt.show()"),
    ("markdown", "## Takeaways\n\n相同总电荷的球对称电荷模型具有相同外场；严格中性、完整同心的界面层没有外场。穿滴电子仍可能感受到内部场与材料散射。这里的连续介质场不等于单个分子键方向的瞬时场。"),
])

notebook("02_transport_validation.ipynb", [
    ("markdown", "# 电子源、轨迹与 Geant4 水滴输运\n\n## tl;dr\n对同一束流比较无材料轨迹和水滴输运，检查成像几何以及进入液滴后的探测分布。"),
    ("markdown", "## Context & Methods\n\n几何尺寸、束腰、能散与发散角来自示例配置；水滴中的能损和散射采用 Geant4 `G4EmStandardPhysics_option4`。此 notebook 采用少量电子作可重跑的功能演示，不用于估计最终检出限。"),
    ("code", setup),
    ("code", "# 两组只改变输运引擎：ray 无材料散射，Geant4 计算水中相互作用。\nfrom droplet_shadow import load_config, simulate\nfrom droplet_shadow.simulation import compare_result\nray = load_config(ROOT / 'examples/baseline-ray.yaml').with_updates(run={'n_simulated': 600, 'exposure_electrons': 600})\nwater = load_config(ROOT / 'examples/baseline-geant4.yaml').with_updates(run={'n_simulated': 600, 'exposure_electrons': 600})\nreference = simulate(ray, ROOT / 'results/notebook_ray')\ntransport = simulate(water, ROOT / 'results/notebook_geant4')\nprint('M=',water.geometry.magnification, 'water hits=',len(transport.hits['x_m']), 'entered fraction=',transport.metrics['entered_detector_fraction'])"),
    ("markdown", "## Results"),
    ("code", "fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)\nfor ax, result, title in zip(axes, [reference, transport], ['Vacuum rays', 'Geant4 water droplet']):\n    ax.hist2d(result.hits['x_m'] * 1e3, result.hits['y_m'] * 1e3, bins=70, range=[[-6,6],[-6,6]], cmap='magma')\n    ax.set(xlabel='Detector x (mm)', ylabel='Detector y (mm)', title=title)\nfig.tight_layout(); plt.show()"),
    ("markdown", "## Takeaways\n\n水滴材料会改变透射电子及次级电子的分布。不同物理机制的定量区分需采用相同源、曝光和探测器设置的参考计算。"),
])

notebook("03_charge_scan.ipynb", [
    ("markdown", "# 净电荷和电子能量的小规模输运扫描\n\n## tl;dr\n用同一组束流参数比较正负净电荷及两档电子能量；正式大规模扫描使用命令行断点续跑。"),
    ("markdown", "## Context & Methods\n\n此 notebook 每组只发射 300 个电子，展示运行路径与输出格式。正式结果需要增加 `n_simulated`、做独立随机种子复算并检查统计误差。"),
    ("code", setup),
    ("code", "# 六点小扫描只展示参数如何替换和结果如何保存，不足以做灵敏度结论。\nfrom droplet_shadow import load_config, simulate\nbase = load_config(ROOT / 'examples/baseline-geant4.yaml').with_updates(run={'n_simulated': 300, 'exposure_electrons': 300})\nresults = {}\nfor energy in [1, 3]:\n    for q in [-1e7, 0, 1e7]:\n        cfg = base.with_updates(source={'energy_mev': energy}, charge={'q_e': q})\n        results[(energy, q)] = simulate(cfg, ROOT / f'results/notebook_scan_E{energy}_Q{q:+.0f}')\nprint('Completed', len(results), 'Geant4 cases')"),
    ("markdown", "## Results"),
    ("code", "fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)\nfor (energy, q), ax in zip(results, axes.flat):\n    hits = results[(energy, q)].hits\n    ax.hist2d(hits['x_m']*1e3, hits['y_m']*1e3, bins=50, range=[[-4,4],[-4,4]], cmap='magma')\n    ax.set(title=f'{energy} MeV, Q={q:+.0e} e', xlabel='x (mm)', ylabel='y (mm)')\nfig.tight_layout(); plt.show()"),
    ("markdown", "## Takeaways\n\n比较相同能量、几何和探测器条件下的图像。低计数示例只说明计算链路可以运行；不能根据这六幅稀疏图像宣布检出限。"),
])

notebook("04_charge_inference.ipynb", [
    ("markdown", "# 从合成图像拟合净电荷\n\n## tl;dr\n使用不同净电荷的同条件期望图像构建模板，并在独立探测器采样上拟合。"),
    ("markdown", "## Context & Methods\n\n这里的数值只是软件演示，模拟粒子数太少，不足以给出实验检出限。正式判据为零电荷下假阳性率 ≤5%、备择电荷下检出率 ≥95%。"),
    ("code", setup),
    ("code", "# 先生成三个同条件 Q 模板；拟合用无噪声 expected，不用 observed。\nfrom droplet_shadow import load_config, simulate\nfrom droplet_shadow.analysis import fit_charge, calibrated_detection_limit\nfrom droplet_shadow.detector import make_images\nbase = load_config(ROOT / 'examples/baseline-geant4.yaml').with_updates(run={'n_simulated': 400, 'exposure_electrons': 400})\ncharges = [-1e7, 0, 1e7]\nmodels = {q: simulate(base.with_updates(charge={'q_e': q}), ROOT / f'results/notebook_fit_Q{q:+.0f}') for q in charges}\ntemplates = {q: models[q].images['expected'] for q in charges}\n# 使用独立探测器随机种子产生一张待拟合图，避免直接拟合模板自身。\nindependent = make_images(models[1e7].config, models[1e7].hits, np.random.default_rng(7654321))['observed']\nfit = fit_charge(independent, templates)\nprint('Known Q=1e7 e; fitted grid Q=', fit.q_e, 'scores=', fit.scores)"),
    ("markdown", "## Results"),
    ("code", "fig, ax = plt.subplots()\nax.plot(charges, [fit.scores[q] for q in charges], marker='o', color='#214e7a')\nax.set(xlabel='Template charge Q (e)', ylabel='Fit score (lower is better)', title='Charge-template comparison')\nfig.tight_layout(); plt.show()"),
    ("markdown", "### 检出限伪实验链路测试\n\n下面每种真实电荷只做 4 次独立束流/输运/探测器采样，仅检查校准算法能运行；不把所得检出率用于实验设计。"),
    ("code", "# 每个真实 Q 重抽 4 次完整曝光：前 2 次设阈值，后 2 次留出验证。\n# 4 次只用于检查软件链路；5% 尾概率和 95% 检出率需要更多重复。\nfrom droplet_shadow.sensitivity import detection_calibration\npaths = {q: ROOT / f'results/notebook_fit_Q{q:+.0f}' for q in charges}\ncalibration = detection_calibration(base.with_updates(run={'n_simulated':100, 'exposure_electrons':100}),\n                                    paths, ROOT / 'results/notebook_limit_demo',\n                                    repetitions=4, fit_nuisance=False)\nprint('Null threshold:', calibration.threshold, 'empirical FPR:', calibration.false_positive_rate)\nprint('Power by Q:', calibration.power_by_charge, 'minimum |Q|:', calibration.minimum_detectable_abs_q_e)\nprint('Charge-grid bias:', calibration.charge_bias_by_true_q_e)\nprint('Held-out interval coverage:', calibration.interval_coverage_by_true_q_e)"),
    ("markdown", "## Takeaways\n\n电荷推断必须和相同材料、源与探测器条件下的图像比较。真正的最低可探测电荷需要高统计量独立模板和至少约 100 次独立曝光，并以经验阈值校准假阳性率。"),
])

print(f"Generated four notebooks in {OUTPUT}")
