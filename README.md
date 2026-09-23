# 单液滴电子 shadow 成像模拟

用解析电场、球对称 Poisson–Boltzmann 模型和 Geant4 电子输运，研究水滴净电荷及界面层对 point-projection shadow 的影响。正式输运使用 `geant4` 引擎；`ray` 引擎用于几何与弱偏转校验。

项目默认把液滴设为静态、半径 50 μm、外部真空，电子经过 crossover 后不再受磁场作用。测得的分子局域界面场不会被当作延伸至探测器的真空 Coulomb 场。参数来源及适用条件见 [文献参数表](data/literature_parameters.csv)与[模型假设说明](docs/model_assumptions.md)。

## 安装与运行

在 Apple Silicon macOS 上，安装 Geant4 11.4.2、CMake 和 Python 依赖：

```bash
conda create -p .conda-env -c conda-forge --override-channels -y \
  python=3.11 pip setuptools wheel geant4=11.4.2 cmake ninja expat zlib numpy scipy matplotlib \
  pyyaml pytest nbformat nbclient ipykernel
conda activate ./.conda-env
PIP_NO_INDEX=1 python -m pip install -e . --no-deps --no-build-isolation
cmake -S cpp -B build -G Ninja -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DEXPAT_INCLUDE_DIR="$CONDA_PREFIX/include" \
  -DEXPAT_LIBRARY="$CONDA_PREFIX/lib/libexpat.dylib" \
  -DZLIB_INCLUDE_DIR="$CONDA_PREFIX/include" \
  -DZLIB_LIBRARY="$CONDA_PREFIX/lib/libz.dylib"
cmake --build build
droplet-shadow simulate --config examples/baseline-geant4.yaml --output results/baseline
```

`results/baseline/` 保存探测器图像、粒子落点、液滴出口能量与方向、运行参数、源代码/可执行文件哈希、指标和 PNG/SVG 图。对同一束流运行带电滴、中性滴、无滴对照：

```bash
droplet-shadow simulate --config examples/baseline-geant4.yaml \
  --output results/controls --controls
```

分析两次运行的期望图像，或对同条件模板拟合电荷：

```bash
droplet-shadow analyze --result results/controls/charged \
  --reference results/controls/neutral
droplet-shadow fit-charge --observed results/controls/charged \
  --template 0 results/controls/neutral \
  --template 1000000 results/controls/charged
```

需要独立重复曝光校准检出限时，先用比伪实验更高的输运样本数生成同条件模板，再运行：

```bash
droplet-shadow estimate-limit --config examples/baseline-geant4.yaml \
  --template 0 results/templates/q0 \
  --template 1000000 results/templates/q1e6 \
  --repetitions 100 --output results/charge-limit
```

此命令保存每次完整输运和探测器采样、拟合分数、5% 假阳性率阈值及独立验证半样本上的检出率、电荷网格偏差与置信区间覆盖率。模板的粒子数不能太低，否则模板 Monte Carlo 噪声会污染阈值；小于 100 次重复仅作软件演示。

按预设能量、电荷、距离及源尺寸扫描。已有且配置一致的单点结果会在再次运行时复用：

```bash
droplet-shadow scan --config examples/baseline-geant4.yaml \
  --output results/scan --n-simulated 10000
```

## 快速检查电场与几何

解析射线引擎不依赖 Geant4，适合检查放大率、有限束腰和 Coulomb 偏转：

```bash
droplet-shadow simulate --config examples/baseline-ray.yaml \
  --output results/ray-check
python -m pytest -q tests
```

`ray` 不包含液滴材料散射，也不能可靠解析厚度为纳米的界面层；涉及穿滴电子或界面层成像的结果应使用 Geant4。

### 输运验证

```bash
python tools/validate_stopping.py --n 5000
python tools/validate_transport.py --n 3000
```

[NIST ESTAR 液态水表](https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html)在 3 MeV 给出总质量阻止本领 1.889 MeV·cm²/g。对 100 μm 水路径，连续减速近似的平均损失为 18.89 keV；当前 5,000 条铅笔束 Geant4 初级电子在水滴出口的平均损失为 19.10 keV。两个脚本保存逐粒子数据、JSON 检查结果以及散射角分布图。出口统计不受探测器接收孔径选择偏差影响；能损涨落及轫致辐射会使中位数与平均数不同。生产截断长度和水滴内部步长可通过 `run.geant4_cut_um`、`run.geant4_core_step_um` 调整。

## 电场模型与成像假设

配置按 `geometry`、`charge`、`source`、`detector`、`run` 分组，长度和能量字段在名称中标明单位。`charge.model` 可选 `surface`、`volume`、`neutral_double_layer`、`poisson_boltzmann`。`poisson_boltzmann` 要求 `q_e = surface_fixed_e + ion_positive_count - ion_negative_count`；离子总数在数值求解中守恒。可追加有限厚度的偶极层及表面电荷的球谐非对称分量。

真实束流采用 `(x,y,x′,y′)` 协方差抽样，横向发射度由协方差确定。图像从输运估计的入射强度生成：探测效率与 MCP 增益进行一次复合 Poisson 抽样，再卷积荧光屏 PSF 并加入像素背景/读出噪声。`n_simulated` 控制输运 Monte Carlo 的样本数；`exposure_electrons` 控制合成曝光，二者分开记录。若前者小于后者，图像会保留输运采样纹理，不能将稀疏图像的环或像素统计视为实验精度；运行记录会标注该警告。

`Q` 的估计依赖同条件模板，并拟合通量、背景、中心位置与额外模糊。正式检出限需使用独立输运随机种子和多次探测器伪实验，以零电荷假阳性率 5%、备择检出率 95% 为判据。少量粒子演示图不能用来声称实验检出限。Geant4 在 20 mm、2 mm 及水滴附近使用分区步长；`run.geant4_*_step_*` 可用于收敛测试。

均匀球壳与均匀体电荷的外部场相同；严格球对称的中性双层在外部没有静电场。穿过水滴的电子仍可能受到内部电场与散射影响。非对称模型对应固定或暂时冻结的表面分布，不假定孤立理想导体处于这种静态分布。

## Notebook

四个 Notebook 依次展示文献场模型、输运对照、电荷扫描和拟合方法。生成并从头执行：

```bash
python tools/make_notebooks.py
python tools/execute_notebooks.py
```

正式研究应先用 `simulate --controls` 检查材料贡献，再提高每组输运粒子数、增加重复随机种子，并检查界面步长与 Geant4 生产截断的收敛性。水滴受束流改变电荷的估计保存在运行指标中；当单次曝光显著改变原设定净电荷时，静态场假设不成立。检出限与非对称性均随具体源、曝光和探测器假设而变，不能从单张图像唯一反演三维电荷分布。

## 参考来源

文献参数表记录 DOI、原始 PDF、页码、数值、介质、尺度及使用限制。Geant4 使用 [11.4 物理列表说明](https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/electromagnetic/Opt4.html)，电子能损核对采用 [NIST ESTAR](https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html)。
