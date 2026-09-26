"""五个教学与审计 Notebook 的源模板。

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
    ("code", "# 只沿 x 轴取样；球对称模型在任何方位的径向场都相同。\n# 宽范围网格用于查看整体尺度，但不解析纳米厚度的界面层。\nR_um = base.geometry.radius_um\nr_wide_um = np.linspace(0.1, 500, 2000)\npoints_wide = np.column_stack((r_wide_um * 1e-6, np.zeros_like(r_wide_um), np.zeros_like(r_wide_um)))\n\n# 在 R 和两个内侧界面附近加入 ±20 nm 的密集网格。\n# 这里 1 nm = 0.001 μm，0.5 nm = 0.0005 μm。\nlayer_edges_um = [R_um, R_um - 1.0e-3, R_um - 0.5e-3]\nlocal_grids_um = [np.linspace(edge - 0.02, edge + 0.02, 801) for edge in layer_edges_um]\nr_zoom_um = np.unique(np.concatenate(local_grids_um))\npoints_zoom = np.column_stack((r_zoom_um * 1e-6, np.zeros_like(r_zoom_um), np.zeros_like(r_zoom_um)))\n\n# 固定净 Q 的表面/体电荷，与零净 Q 的界面层形成对照。\nmodels = {'surface': base, 'volume': base.with_updates(charge={'model': 'volume'}),\n          'neutral double layer': base.with_updates(charge={'model': 'neutral_double_layer', 'q_e': 0, 'double_layer_q_e': 1e8}),\n          '1 V dipole layer': base.with_updates(charge={'model': 'surface', 'q_e': 0, 'dipole_potential_v': 1.0})}\n\n# 每个模型单独出图；两个坐标轴都保持线性，避免对数轴掩盖边界跳变。\nfor name, cfg in models.items():\n    f = build_field(cfg)\n    field_wide_mvm = abs(f.electric_field(points_wide)[:, 0]) / 1e6\n    field_zoom_mvm = abs(f.electric_field(points_zoom)[:, 0]) / 1e6\n    zoom = (r_zoom_um >= R_um - 0.02) & (r_zoom_um <= R_um + 0.02)\n\n    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n    axes[0].plot(r_wide_um, field_wide_mvm, color='#214e7a')\n    axes[0].axvline(R_um, color='0.35', linestyle='--', linewidth=1, label='droplet boundary')\n    axes[0].set(xlabel='Radius (μm)', ylabel='|E| (MV/m)',\n                title=f'{name}: full radial range')\n    axes[0].legend()\n\n    axes[1].plot((r_zoom_um[zoom] - R_um) * 1e3, field_zoom_mvm[zoom], color='#c23b22')\n    axes[1].axvline(0, color='0.35', linestyle='--', linewidth=1, label='r = R')\n    axes[1].set(xlabel='r − R (nm)', ylabel='|E| (MV/m)',\n                title=f'{name}: interface zoom')\n    axes[1].legend()\n    fig.tight_layout(); plt.show()"),
    ("markdown", "### 守恒离子数的球对称 Poisson–Boltzmann 解"),
    ("code", "# -10^6 e 固定表面电荷与 +10^6 个移动反离子相抵：总净 Q=0。\nfrom droplet_shadow.fields import poisson_boltzmann\nneutral = base.with_updates(charge={'model':'poisson_boltzmann', 'q_e':0, 'surface_fixed_e':-1e6, 'ion_positive_count':1e6})\npb = poisson_boltzmann(neutral)\n# 100 μm > R=50 μm；严格球对称且净电荷为零时，外场应为零。\nprint('External E for neutral PB (V/m):', pb.electric_field(np.array([[100e-6,0,0]]))[0,0])\nfig, ax = plt.subplots()\nax.plot(pb.radii_m * 1e6, pb.field_v_m / 1e6, color='#214e7a')\nax.set(xlabel='Radius (μm)', ylabel='Internal radial E (MV/m)', title='Neutral fixed surface charge with mobile counterions')\nfig.tight_layout(); plt.show()"),
    ("markdown", "## 实现追溯\n\n本 Notebook 调用的构造器在 [`fields.py`](../src/droplet_shadow/fields.py)，参数约束在 [`config.py`](../src/droplet_shadow/config.py)，可执行物理不变量在 [`test_physics.py`](../tests/test_physics.py)。其中 `RadialShells._radial`、`UniformVolume._radial`和 `poisson_boltzmann` 是人工复核的主要入口。更完整的逐层审计见 `05_implementation_audit.ipynb`。"),
    ("markdown", "## Takeaways\n\n每个模型现在单独显示，两个坐标轴均为线性。`volume` 模型在液滴外部应是一条平滑的 1/r² 曲线；在 r=R 处的电场跳变来自水/真空介电边界。严格中性双层和偶极层在外部没有宏观电场，但在纳米界面放大图中仍可看到局域场。穿滴电子仍可能感受到内部场与材料散射。这里的连续介质场不等于单个分子键方向的瞬时场。"),
])

notebook("02_transport_validation.ipynb", [
    ("markdown", "# 电子源、轨迹与 Geant4 水滴输运\n\n## tl;dr\n对同一束流比较无材料轨迹和水滴输运，检查成像几何以及进入液滴后的探测分布。"),
    ("markdown", "## Context & Methods\n\n几何尺寸、束腰、能散与发散角来自示例配置；水滴中的能损和散射采用 Geant4 `G4EmStandardPhysics_option4`。此 notebook 采用少量电子作可重跑的功能演示，不用于估计最终检出限。"),
    ("code", setup),
    ("code", "# 两组只改变输运引擎：ray 无材料散射，Geant4 计算水中相互作用。\nfrom droplet_shadow import load_config, simulate\nfrom droplet_shadow.simulation import compare_result\nray = load_config(ROOT / 'examples/baseline-ray.yaml').with_updates(run={'n_simulated': 600, 'exposure_electrons': 600})\nwater = load_config(ROOT / 'examples/baseline-geant4.yaml').with_updates(run={'n_simulated': 600, 'exposure_electrons': 600})\nreference = simulate(ray, ROOT / 'results/notebook_ray')\ntransport = simulate(water, ROOT / 'results/notebook_geant4')\nprint('M=',water.geometry.magnification, 'water hits=',len(transport.hits['x_m']), 'entered fraction=',transport.metrics['entered_detector_fraction'])"),
    ("markdown", "## Results"),
    ("code", "fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)\nfor ax, result, title in zip(axes, [reference, transport], ['Vacuum rays', 'Geant4 water droplet']):\n    ax.hist2d(result.hits['x_m'] * 1e3, result.hits['y_m'] * 1e3, bins=70, range=[[-6,6],[-6,6]], cmap='magma')\n    ax.set(xlabel='Detector x (mm)', ylabel='Detector y (mm)', title=title)\nfig.tight_layout(); plt.show()"),
    ("markdown", "## 实现追溯\n\n电子源协方差与相对论动量在 [`source.py`](../src/droplet_shadow/source.py)，无材料轨迹在 [`rays.py`](../src/droplet_shadow/rays.py)，Python/Geant4 单位转换与 CSV 接口在 [`native.py`](../src/droplet_shadow/native.py)，Geant4 物理、步长和记录逻辑在 [`cpp/main.cpp`](../cpp/main.cpp)。详细中间量见第 5 个 Notebook。"),
    ("markdown", "## Takeaways\n\n水滴材料会改变透射电子及次级电子的分布。不同物理机制的定量区分需采用相同源、曝光和探测器设置的参考计算。"),
])

notebook("03_charge_scan.ipynb", [
    ("markdown", "# 净电荷和电子能量的小规模输运扫描\n\n## tl;dr\n用同一组束流参数比较正负净电荷及两档电子能量；正式大规模扫描使用命令行断点续跑。"),
    ("markdown", "## Context & Methods\n\n此 notebook 每组只发射 300 个电子，展示运行路径与输出格式。正式结果需要增加 `n_simulated`、做独立随机种子复算并检查统计误差。"),
    ("code", setup),
    ("code", "# 六点小扫描只展示参数如何替换和结果如何保存，不足以做灵敏度结论。\nfrom droplet_shadow import load_config, simulate\nbase = load_config(ROOT / 'examples/baseline-geant4.yaml').with_updates(run={'n_simulated': 300, 'exposure_electrons': 300})\nresults = {}\nfor energy in [1, 3]:\n    for q in [-1e7, 0, 1e7]:\n        cfg = base.with_updates(source={'energy_mev': energy}, charge={'q_e': q})\n        results[(energy, q)] = simulate(cfg, ROOT / f'results/notebook_scan_E{energy}_Q{q:+.0f}')\nprint('Completed', len(results), 'Geant4 cases')"),
    ("markdown", "## Results"),
    ("code", "fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)\nfor (energy, q), ax in zip(results, axes.flat):\n    hits = results[(energy, q)].hits\n    ax.hist2d(hits['x_m']*1e3, hits['y_m']*1e3, bins=50, range=[[-4,4],[-4,4]], cmap='magma')\n    ax.set(title=f'{energy} MeV, Q={q:+.0e} e', xlabel='x (mm)', ylabel='y (mm)')\nfig.tight_layout(); plt.show()"),
    ("markdown", "## 实现追溯\n\n单次运行、结果保存与版本哈希在 [`simulation.py`](../src/droplet_shadow/simulation.py)，扫描网格、缓存命中和断点续跑在 [`cli.py`](../src/droplet_shadow/cli.py)。人工检查时应同时打开每个结果目录的 `result.json`、`hits.npz`和 `images.npz`。"),
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
    ("markdown", "## 实现追溯\n\n图像生成在 [`detector.py`](../src/droplet_shadow/detector.py)，模板偏差与扰动参数拟合在 [`analysis.py`](../src/droplet_shadow/analysis.py)，校准/留出分割在 [`sensitivity.py`](../src/droplet_shadow/sensitivity.py)。请特别复核 `make_images`的噪声顺序、`fit_charge`的加权残差评分，以及 `detection_calibration`是否将同一曝光同时用于定阈值和报告检出率。"),
    ("markdown", "## Takeaways\n\n电荷推断必须和相同材料、源与探测器条件下的图像比较。真正的最低可探测电荷需要高统计量独立模板和至少约 100 次独立曝光，并以经验阈值校准假阳性率。"),
])

notebook("05_implementation_audit.ipynb", [
    ("markdown", "# 实现审计：从配置到原始结果\n\n## tl;dr\n\n这本 Notebook 不是第五个物理结果演示，而是人工代码复核的导航和可执行检查表。它展示中间量、单位转换、数值不变量、Geant4 输入文件和结果溯源记录。所有 `assert` 都通过才能执行到最后。"),
    ("markdown", "## 1. 源码地图\n\n建议按数据流阅读：\n\n1. [`config.py`](../src/droplet_shadow/config.py)：参数、单位和非法组合。\n2. [`source.py`](../src/droplet_shadow/source.py)：四维高斯相空间与相对论动量。\n3. [`fields.py`](../src/droplet_shadow/fields.py)：球壳、体电荷、双层、PB 与共享的径向介电层。\n4. [`rays.py`](../src/droplet_shadow/rays.py)：无材料相对论轨迹。\n5. [`native.py`](../src/droplet_shadow/native.py) → [`cpp/main.cpp`](../cpp/main.cpp)：Python/Geant4 单位边界和完整材料输运。\n6. [`detector.py`](../src/droplet_shadow/detector.py)：落点到理想/期望/观测图。\n7. [`analysis.py`](../src/droplet_shadow/analysis.py) 与 [`sensitivity.py`](../src/droplet_shadow/sensitivity.py)：特征、拟合与检出限校准。\n8. [`simulation.py`](../src/droplet_shadow/simulation.py) 与 [`cli.py`](../src/droplet_shadow/cli.py)：管线、输出、版本指纹、扫描缓存。\n9. [`tests/`](../tests)：自动化验收边界；测试通过不等于物理假设已被实验证实。"),
    ("code", setup),
    ("code", "# 不复制整个源文件；这个工具显示任意 Python 实现的真实文件、起始行和源码。\nimport inspect, json\nfrom pprint import pprint\n\ndef show_implementation(obj):\n    lines, first = inspect.getsourcelines(obj)\n    path = Path(inspect.getsourcefile(obj)).resolve()\n    print(f'{path.relative_to(ROOT)}:{first}-{first + len(lines) - 1}')\n    print(''.join(f'{first+i:4d}  {line}' for i, line in enumerate(lines)))"),
    ("markdown", "## 2. 配置、单位与不可变性\n\nYAML 对人使用 μm/mm/MeV；进入电场和 Python 轨迹后统一为 SI。`with_updates` 返回新对象，不会暗中修改扫描基准。"),
    ("code", "from droplet_shadow import SimulationConfig, load_config, build_field, simulate\nfrom droplet_shadow.config import Geometry\nbase = load_config(ROOT / 'examples/baseline-ray.yaml')\nchanged = base.with_updates(charge={'q_e': 1e7})\nassert base.charge.q_e != changed.charge.q_e\nassert np.isclose(base.geometry.radius_m, 50e-6)\nassert np.isclose(base.geometry.magnification, 51.0)\nprint('radius_m =', base.geometry.radius_m, 'magnification =', base.geometry.magnification)\nprint('base Q, changed Q =', base.charge.q_e, changed.charge.q_e)\ntry:\n    Geometry(radius_um=-1)\nexcept ValueError as error:\n    print('invalid geometry correctly rejected:', error)\nshow_implementation(SimulationConfig.with_updates)"),
    ("markdown", "## 3. 真实电子源的协方差\n\n前两列是 crossover 位置 (m)，后两列是斜率 `dx/dz, dy/dz`。FWHM 先除以 2.35482 转为 RMS；发射度是 2×2 协方差行列式平方根，不是另一个自由参数。"),
    ("code", "from droplet_shadow.source import phase_space_covariance, geometric_emittance, sample_source\ncov = phase_space_covariance(base.source)\nphase, momentum = sample_source(base.source, 100_000, np.random.default_rng(20260923))\nempirical = np.cov(phase, rowvar=False)\nprint('configured covariance:')\nprint(cov)\nprint('empirical/configured diagonal ratio:', np.diag(empirical) / np.diag(cov))\nprint('geometric emittance x,y (m rad):', geometric_emittance(base.source))\nnp.testing.assert_allclose(np.diag(empirical), np.diag(cov), rtol=0.02)\nshow_implementation(phase_space_covariance)\nshow_implementation(sample_source)"),
    ("markdown", "## 4. 电场模型的可执行不变量\n\n这里不只画曲线，而是直接计算外场简并、内场差异、双层外场和偶极层势差。断言的容差是数值精度，不是实验不确定度。"),
    ("code", "qcfg = SimulationConfig().with_updates(charge={'q_e': 1e6})\nR = qcfg.geometry.radius_m\noutside = np.array([[2*R, 0, 0], [4*R, 0, 0]])\ninside = np.array([[0.2*R, 0, 0]])\nsurface = build_field(qcfg)\nvolume = build_field(qcfg.with_updates(charge={'model':'volume'}))\nexternal_rel_error = np.max(abs(surface.electric_field(outside)-volume.electric_field(outside))) / np.max(abs(surface.electric_field(outside)))\nprint('surface/volume maximum exterior relative error =', external_rel_error)\nprint('interior Ex surface, volume (V/m) =', surface.electric_field(inside)[0,0], volume.electric_field(inside)[0,0])\nassert external_rel_error < 1e-12\nassert not np.allclose(surface.electric_field(inside), volume.electric_field(inside))\n\ndlcfg = qcfg.with_updates(charge={'model':'neutral_double_layer', 'q_e':0, 'double_layer_q_e':1e8})\ndl = build_field(dlcfg)\nprint('neutral double-layer exterior max |E| (V/m) =', np.max(abs(dl.electric_field(outside))))\nnp.testing.assert_allclose(dl.electric_field(outside), 0, atol=1e-10)\n\ndpcfg = qcfg.with_updates(charge={'model':'surface', 'q_e':0, 'dipole_potential_v':1.0})\ndipole = build_field(dpcfg)\ndelta_phi = dipole.potential(np.array([[0,0,0]]))[0] - dipole.potential(outside[:1])[0]\nprint('1 V layer center-to-exterior potential difference (V) =', delta_phi)\nassert abs(delta_phi - 1.0) < 1e-9"),
    ("code", "# 球对称 PB：-10^6 e 固定面电荷 + 10^6 个可移动正离子。\nfrom droplet_shadow.fields import poisson_boltzmann, RadialShells, UniformVolume\npbcfg = qcfg.with_updates(charge={'model':'poisson_boltzmann', 'q_e':0, 'surface_fixed_e':-1e6, 'ion_positive_count':1e6})\npb = poisson_boltzmann(pbcfg)\npb_external = pb.electric_field(np.array([[2*R,0,0]]))[0,0]\nprint('neutral PB exterior Ex (V/m) =', pb_external)\nprint('PB nodes =', len(pb.radii_m), '; surface-side E (MV/m) =', pb.field_v_m[-1]/1e6)\nassert abs(pb_external) < 1e-9\nshow_implementation(RadialShells._radial)\nshow_implementation(UniformVolume._radial)\nshow_implementation(poisson_boltzmann)"),
    ('markdown', '### 球对称修复的数值检查\n\nPB 现在以表面势为参考求解，再加回绝对表面电势，避免高净电荷导致归一化指数相消。网格包含精确 R；原始表末行是内侧场，查询 R 则统一取外侧场。电势采用 Hermite 插值，电场从同一插值取负导数。'),
    ('code', "assert pb.radii_m[-1] == R\nprint('PB exact outer radius (m):', pb.radii_m[-1])\nprint('PB conserved ion counts:', pb.ion_counts)\nprint('PB E(R-) from table, E(R+) by query (V/m):', pb.field_v_m[-1], pb.electric_field(np.array([[R,0,0]]))[0,0])\nnp.testing.assert_allclose(pb.ion_counts, [1e6], rtol=1e-9)\nr_check = R*np.array([0.2,0.5,0.9])\np_check = np.column_stack((r_check,np.zeros(3),np.zeros(3)))\nh = R*1e-6\nminus_gradient = -(pb.potential(p_check+[h,0,0])-pb.potential(p_check-[h,0,0]))/(2*h)\nnp.testing.assert_allclose(pb.electric_field(p_check)[:,0], minus_gradient, rtol=2e-5, atol=1e-5)\nprint('PB field/potential derivative agreement: passed')\n# 体电荷与偶极层共享介电分布：水核 78，外层 2。\ncombined = build_field(qcfg.with_updates(charge={'model':'volume','dipole_potential_v':1.0}))\nprint('Shared dielectric profiles:', [(f.epsilon_inside,f.layer_inner_m,f.epsilon_layer) for f in combined.fields])\nassert len({(f.layer_inner_m,f.epsilon_layer) for f in combined.fields}) == 1"),
    ("markdown", "## 5. 无材料轨迹与弱偏转对照\n\n用一个入射参数 100 μm、1 MeV 的电子检查偏转方向和量级。正液滴吸引电子，所以最终 x 小于无场值。"),
    ("code", "from droplet_shadow.rays import trace_rays, weak_deflection_angle, z_mesh\nfrom droplet_shadow.constants import C, E_CHARGE, M_E\nraycfg = SimulationConfig().with_updates(source={'energy_mev':1.0}, charge={'q_e':1e6}, run={'engine':'ray','n_simulated':1,'exposure_electrons':1})\nb = 100e-6\nK = 1e6 * E_CHARGE\np = np.sqrt(K*(K+2*M_E*C**2))/C\none = trace_rays(raycfg, build_field(raycfg), np.array([[b,0,0,0]]), np.array([[0,0,p]]))\nalpha = weak_deflection_angle(1e6, b, 1.0)\nnumeric_dx = one['x_m'][0]-b\nanalytic_dx = raycfg.geometry.l2_mm*1e-3*alpha\nprint('z mesh points =', len(z_mesh(raycfg)))\nprint('numeric dx, weak-deflection dx (m) =', numeric_dx, analytic_dx)\nprint('ratio =', numeric_dx/analytic_dx)\nassert numeric_dx < 0 and abs(numeric_dx/analytic_dx-1) < 0.15\nshow_implementation(trace_rays)"),
    ("markdown", "## 6. Python → Geant4 文件边界\n\n本节只导出小样本，不隐式启动耗时的 Geant4 计算。`source.csv` 是 mm/斜率/MeV，`field.csv` 是 mm/V/V·mm⁻¹。界面壳前后会加额外采样点，以防插值把跃变抹平。"),
    ("code", "from droplet_shadow.native import export_source, export_field\nfrom droplet_shadow.source import sample_source\naudit_native = ROOT / 'results/notebook_implementation_audit/spherical_native_inputs'\naudit_native.mkdir(parents=True, exist_ok=True)\nsmall_phase, small_p = sample_source(base.source, 16, np.random.default_rng(7))\nexport_source(dlcfg, small_phase, small_p, audit_native/'source.csv')\nexport_field(dlcfg, audit_native/'field.csv')\nfor filename in ['source.csv','field.csv']:\n    path = audit_native/filename\n    lines = path.read_text().splitlines()\n    print(f'\\n{path.relative_to(ROOT)}: {len(lines)-1} data rows')\n    print('\\n'.join(lines[:4]))\nassert (audit_native/'source.csv').read_text().splitlines()[0] == 'x_mm,y_mm,xprime,yprime,energy_MeV'\nassert (audit_native/'field.csv').read_text().splitlines()[0] == 'radius_mm,potential_v,field_v_mm'\nshow_implementation(export_field)"),
    ("code", "# 从 C++ 中抽出关键实现行；完整文件请点击上方 cpp/main.cpp。\ncpp = (ROOT/'cpp/main.cpp').read_text().splitlines()\nkeywords = ('G4EmStandardPhysics_option4', 'G4EqMagElectricField', 'G4ClassicalRK4', 'SetUserAction', 'hits.csv', 'deposition.csv')\nfor number, line in enumerate(cpp, 1):\n    if any(word in line for word in keywords):\n        print(f'{number:4d}  {line}')"),
    ("markdown", "## 7. 落点 → 探测器三层图像\n\n`ideal` 是按曝光重权累加的落点；`expected` 加入效率、平均增益和 PSF，但没有这次曝光的随机噪声；`observed` 只做一次复合 Poisson/增益/背景/读出抽样。"),
    ("code", "from droplet_shadow.detector import make_images\nimage_cfg = base.with_updates(charge={'q_e':1e6}, run={'engine':'ray','n_simulated':5000,'exposure_electrons':10000,'seed':314159})\nphase_i, p_i = sample_source(image_cfg.source, image_cfg.run.n_simulated, np.random.default_rng(image_cfg.run.seed))\nhits_i = trace_rays(image_cfg, build_field(image_cfg), phase_i, p_i)\nimages = make_images(image_cfg, hits_i, np.random.default_rng(image_cfg.run.seed+1))\nprint('image shape =', images['ideal'].shape)\nprint('sum ideal, expected, observed =', *(images[k].sum() for k in ['ideal','expected','observed']))\nassert images['ideal'].shape == images['expected'].shape == images['observed'].shape\nfig, axes = plt.subplots(1,3,figsize=(13,4))\nfor ax, key in zip(axes, ['ideal','expected','observed']):\n    ax.imshow(images[key], origin='lower', cmap='magma')\n    ax.set(title=key, xlabel='pixel x', ylabel='pixel y')\nfig.tight_layout(); plt.show()\nshow_implementation(make_images)"),
    ("markdown", "## 8. 管线输出与版本溯源\n\n一次运行应保存原始落点、三层图像、完整参数、软件版本和源码 SHA-256。这样人工复核可区分“配置改了”和“实现改了”。"),
    ("code", "audit_result_dir = ROOT/'results/notebook_implementation_audit/small_ray_result'\nresult = simulate(image_cfg.with_updates(run={'n_simulated':300,'exposure_electrons':300}), audit_result_dir)\nrecord = json.loads((audit_result_dir/'result.json').read_text())\nprint('saved files:', sorted(p.name for p in audit_result_dir.iterdir()))\nprint('source SHA-256:', record['versions']['source_sha256'])\nprint('metric names:', sorted(record['metrics']))\nassert {'images.npz','hits.npz','result.json','shadow.png','shadow.svg'} <= {p.name for p in audit_result_dir.iterdir()}\nassert record['config']['run']['n_simulated'] == 300"),
    ("markdown", "## 9. 拟合、检出限与人工检查清单\n\n继续人工阅读：[`fit_charge`](../src/droplet_shadow/analysis.py) 同时处理电荷网格和通量/背景/中心/PSF 扰动参数；[`detection_calibration`](../src/droplet_shadow/sensitivity.py) 使用分离的校准与留出曝光。\n\n手工验收时建议逐项记录：\n\n- [ ] 每个 YAML 数值的物理来源或“情景假设”标签。\n- [ ] 配置单位 → SI → Geant4 单位的每一个转换。\n- [ ] 电场边界条件、总电荷和球外场的解析对照。\n- [ ] 界面网格、轨迹容差、Geant4 步长与 production cut 收敛。\n- [ ] `n_simulated` 与 `exposure_electrons` 分开，且观测图未被二次加 shot noise。\n- [ ] 模板与待拟合曝光的随机种子独立。\n- [ ] 5% 假阳性阈值和 95% 检出率来自留出数据，重复次数足够。\n- [ ] 对照 NIST 能损、散射分布及实验仪器标定。\n\n## Takeaways\n\n这本 Notebook 公开了从参数到原始结果的主要中间层，但不代替逐行 code review。它验证的是实现内部一致性，不是未标定仪器的绝对准确度，也不能将连续介质偶极层解释成单分子瞬时电场。"),
])

print(f"Generated five notebooks in {OUTPUT}")
