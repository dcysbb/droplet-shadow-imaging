# 单液滴电子 shadow 成像模拟

用解析电场、球对称 Poisson–Boltzmann 模型和 Geant4 电子输运，研究水滴净电荷及界面层对 point-projection shadow 的影响。正式输运使用 `geant4` 引擎；`ray` 引擎用于几何与弱偏转校验。

项目默认把液滴设为静态、半径 50 μm、外部真空，电子经过 crossover 后不再受磁场作用。测得的分子局域界面场不会被当作延伸至探测器的真空 Coulomb 场。参数来源及适用条件见 [文献参数表](data/literature_parameters.csv)与[模型假设说明](docs/model_assumptions.md)。

## 安装与运行

### 本轮研究：Linux 迁移、35 组配置、PSF=0

新增的 [Linux 运行说明](docs/linux_field_study.md) 对应本轮三类电场与
10/5/1 μm 焦点对照。[研究清单](examples/field_scale_study/manifest.json)
包含 35 份完整配置，每组 1000 万初级电子，当前均未运行。
`python tools/run_field_scale_study.py` 默认只检查，不启动计算。
人工阅读使用 `notebooks/06_field_scale_comparison.ipynb`，缺失结果不会自动补跑。
本轮关闭 PSF 不改变下面原有示例的 50 μm PSF 默认值。
不要在只检查本轮交付时运行“全部 Notebook”或“全部测试”，其中旧内容可能启动真实输运。

### 原有 macOS 安装方式

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

### 使用多线程而不降低精度

**升级后先执行 `cmake --build build`。** 新的 Python 接口会检查原生协议；旧二进制会提示重编译，而不是用错误的位置参数继续运行。

在已有 YAML 的 `run:` 分组添加（不要重复创建第二个 `run:`）：

```yaml
run:
  engine: geant4
  geant4_threads: 0
  n_simulated: 1000000
  exposure_electrons: 1000000
  raster_width_mm: 40
```

| `geant4_threads` | 行为 |
|---|---|
| `0`（默认） | 可用逻辑 CPU 数减一，至少 1、最多 8 个工作线程 |
| `1` | 串行输运，适合调试和性能基线 |
| 正整数，例如 `4` | 指定工作线程数，不受自动模式的 8 线程上限约束 |

实际线程数不超过本次初级事件数。自动模式在不支持 MT 的 Geant4 构建上警告并退回串行；显式请求多线程则报错。请勿设置 `G4FORCENUMBEROFTHREADS` 覆盖 YAML。`ray` 和 `ideal_occluder` 不使用此参数。扫描、三组对照仍逐个运行，每个 Geant4 单点内部并行；不要让两个运行写入同一输出目录。

**运行进度默认开启。** 在已有 `run:` 分组设置 `show_progress: false` 可关闭，设为 `true` 开启，无需安装额外依赖。升级后须重新编译核心，Notebook 须重启内核。Geant4 输运时显示已完成初级电子数、百分比、平均速度及预计剩余时间；终端原地刷新，Notebook 更新同一个显示项，重定向输出时约每五秒记录一次，避免刷屏。射线引擎目前只显示处理阶段，不提供逐电子实时计数。

百分比的分母是 `n_simulated`，不是曝光量或命中数。输运 100% 不等于所有工作结束，之后会显示“合并与检查事件记录”“生成探测器图像”“保存结果与图像”，成功结束才显示“完成”。预计剩余时间仅估算输运阶段，不能预测后处理；复杂次级事件可能使估算波动。失败/中断不会伪装成成功，中断会终止并回收原生子进程。详细日志仍在 `native/geant4.log`。

进度来自各 worker 完成事件后的原子计数，由限频快照传给 Python 显示，不在每个积分步打印，也不改变随机数抽样。可用 `python -m pytest -q tests/test_progress.py` 验证显示、实时更新、异常处理及进度开关前后的输运结果一致性。

并行单位是**完整初级事件**（包括其次级粒子），并未改变物理列表、场表、积分容差或步长。几何和源/场数据只读共享，事件状态、积分器和输出流线程独立。每个 worker 先写 `native/workers_*/worker_*_*.csv`；所有事件完整性检查通过后，按事件/轨迹编号稳定合并为原有 `hits.csv`、`exits.csv` 和 `deposition.csv`。重复出滴记录不会被去重，沉积电荷按事件编号还原。失败分片会保留供诊断。

Geant4 负责工作线程的独立随机流和逐事件种子分配；同版本、同配置、同线程数和种子的结果可重复。不同线程数只承诺统计一致，不承诺逐粒子相同，也不要求复现旧二进制的随机轨迹。不要把串行与并行的随机差异直接解释为物理差异。

`result.json` 的 `runtime` 保存实际线程数、随机引擎、各 worker 完成事件数，以及源生成、输入准备、初始化、输运、合并读取、探测器处理和保存时间。`native/transport.json` 单独保存输运记录。`transport_s` 是 `BeamOn` 墙钟时间，包含 worker 启动、分片写入及结束同步；不是各线程 CPU 时间之和。`total` 包含保存图像，但不包含最后的小型结果 JSON 写入。

**视野不能无限放大：** 25 μm 像素下，40 mm 视野约为 1600×1600；750 mm 视野则为 30000×30000，单个 float64 数组约 7.2 GB，且后处理同时持有多个数组。程序在输运前对超过一千万像素的配置发出警告，不会自行降低分辨率。当前示例已改为 40 mm，覆盖有效探测器；保留了原有液滴、源和 2000 mm 探测距离。

复现性能与统计检查（每次选择新的输出目录）：

```bash
python tools/benchmark_threads.py --n 10000 --repetitions 3 \
  --output results/thread-performance
python tools/benchmark_threads.py --statistics --n 10000 --repetitions 5 \
  --output results/thread-statistics
```

脚本分别测试表面电荷和 10 nm 中性双层，固定物理精度与 40 mm 视野；性能模式比较 1/2/4/自动线程，统计模式以五个独立种子比较 1/4 线程。输出 `runs.json`、`summary.json` 及各次完整模拟。运行性能测试时不要同时执行其他模拟。少量电子时初始化和图像保存可能主导耗时，多线程不保证更快；提高 `exposure_electrons` 也不能替代提高 `n_simulated`。实测记录见 [多线程验收说明](docs/multithreading_validation.md)。

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

本版本只研究球对称场。配置按 `geometry`、`charge`、`source`、`detector`、`run` 分组，长度和能量字段在名称中标明单位。`charge.model` 可选 `surface`、`volume`、`neutral_double_layer`、`poisson_boltzmann`。`poisson_boltzmann` 要求 `q_e = surface_fixed_e + ion_positive_count - ion_negative_count`；离子总数在数值求解中守恒。固定电荷模型可追加中性双层和均匀径向偶极层。

激活偶极层后，`R-dipole_thickness_nm` 到 `R` 使用 `dipole_epsilon`，内核使用 `epsilon_water`；所有固定电荷分量共用此介电分布。`dipole_potential_v` 是偶极分量自身的势差，组合模型的总势差还包含其他电荷贡献。PB 不能与额外界面层直接相加。

PB 网格包含精确表面 `r=R`，电势采用 Hermite 插值，电场由同一插值取负导数。`field_v_m[-1]` 保存表面内侧场，`electric_field([R,0,0])` 统一返回外侧场。Geant4 场表用同半径的两行保存内、外侧极限，避免插值把真实跳变抹平。`--controls` 的中性参考清除全部自由电荷和界面层，定义为无场水滴。

球谐、斑块和角向分析接口已移除；旧配置中的相关字段需要删除，不能继续解释为有效输入。原生 CSV/命令行接口也已更新，升级源码后须运行 `cmake --build build` 重新构建 Geant4 核心。

真实束流采用 `(x,y,x′,y′)` 协方差抽样，横向发射度由协方差确定。图像从输运估计的入射强度生成：探测效率与 MCP 增益进行一次复合 Poisson 抽样，再卷积荧光屏 PSF 并加入像素背景/读出噪声。`n_simulated` 控制输运 Monte Carlo 的样本数；`exposure_electrons` 控制合成曝光，二者分开记录。若前者小于后者，图像会保留输运采样纹理，不能将稀疏图像的环或像素统计视为实验精度；运行记录会标注该警告。

`Q` 的估计依赖同条件模板，并拟合通量、背景、中心位置与额外模糊。正式检出限需使用独立输运随机种子和多次探测器伪实验，以零电荷假阳性率 5%、备择检出率 95% 为判据。少量粒子演示图不能用来声称实验检出限。Geant4 在 20 mm、2 mm 及水滴附近使用分区步长；`run.geant4_*_step_*` 可用于收敛测试。

均匀球壳与均匀体电荷的外部场相同；严格球对称的中性双层在外部没有静电场。穿过水滴的电子仍可能受到内部电场与散射影响。有限源抽样和散射产生的单次图像可以不完全圆对称，但电场本身严格球对称。

## Notebook

前四个 Notebook 依次展示文献场模型、输运对照、电荷扫描和拟合方法；
`05_implementation_audit.ipynb` 逐层展开配置、单位、源协方差、场不变量、轨迹、Geant4
文件边界、探测器与结果溯源，供人工审查实现。生成并从头执行：

```bash
python tools/make_notebooks.py
python tools/execute_notebooks.py
```

正式研究应先用 `simulate --controls` 检查材料贡献，再提高每组输运粒子数、增加重复随机种子，并检查界面步长与 Geant4 生产截断的收敛性。水滴受束流改变电荷的估计保存在运行指标中；当单次曝光显著改变原设定净电荷时，静态场假设不成立。检出限随源、曝光和探测器假设而变；相同净电荷的不同球内分布也可能无法区分。

## 参考来源

文献参数表记录 DOI、原始 PDF、页码、数值、介质、尺度及使用限制。Geant4 使用 [11.4 物理列表说明](https://geant4.web.cern.ch/documentation/dev/plg_html/PhysicsListGuide/electromagnetic/Opt4.html)，电子能损核对采用 [NIST ESTAR](https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html)。
