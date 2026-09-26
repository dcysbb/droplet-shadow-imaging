# Linux 上运行三类电场与焦点尺寸研究

**交付阶段没有运行任何正式电子输运、Geant4 探针或性能测试。35 组均为未运行。**
配置/代码准备就绪不等于物理输运已通过新机器验收；请先执行下面的验证步骤。
本研究关闭 PSF，保留像素积分、有限源和水中散射。图像是否能够区分模型由人工判断。

## 1. 复制源码并建立 Linux 环境

把 `droplet-shadow-linux-source.tar.gz` 复制到目标电脑，进入希望存放项目的目录后执行：

```bash
tar -xzf droplet-shadow-linux-source.tar.gz
cd droplet-shadow-imaging
conda env create -f environment-linux.yml
conda activate droplet-shadow
python -m pip install -e . --no-deps --no-build-isolation
cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build --parallel 4
build/droplet_g4 --capabilities
geant4-config --version
geant4-config --datasets
```

需要事先安装 Conda/Miniforge。环境文件固定 Python 3.11、Geant4 11.4.2、NumPy 2.4.6、SciPy 1.17.1，
其余依赖由 conda-forge 求解；它不是跨所有平台的逐包锁文件。首次安装需要网络和相应平台的软件包。
若目标架构没有所列版本，先报告求解错误，不要不记录版本就换成其他物理库。

预期能力查询包含协议 `2`、多线程能力和进度支持。该命令只查询能力，不发射电子。
请检查版本确为 11.4.2，数据集目录存在；“编译成功”不代表运行所需物理数据集齐全。
配置中的 `geant4_executable: null` 使用本项目的 `build/droplet_g4`。
不要复制 Mac 的 `.conda-env`、`build`、`.dylib`，也不要把 Mac 绝对路径填入 YAML。

安装依据：[Geant4 官方 Unix/CMake 安装说明](https://geant4.web.cern.ch/documentation/pipelines/master/ig_html/InstallationGuide/installguide.html)、
[conda-forge Geant4 包](https://anaconda.org/conda-forge/geant4)。官方网页示例版本可能不同，本项目仍要求上面固定的版本。
这里给出的是 Linux 运行指令；交付时未在 Linux 上实际安装或编译验证。

## 2. 先做不会启动输运的检查

下面的命令可以安全地检查交付内容，不调用 Geant4 输运，也不调用 Python ray 轨迹积分：

```bash
python tools/prepare_field_study.py
python tools/run_field_scale_study.py
python tools/validate_field_study.py
python tools/check_field_study_notebook.py
python -m pytest -q tests/test_field_study.py
```

`prepare_field_study.py` 比较完整 YAML 与研究定义。若你手动修改 YAML，会报告差异而不是覆盖它。
`run_field_scale_study.py` 默认只列条件和资源估计；`validate_field_study.py` 默认只列探针条件。
轻量测试用禁止真实输运的替身和临时假落点，只检验程序逻辑，不作为研究结果。
`python tools/check_field_study_notebook.py --execute` 只执行新的 06，并临时禁止输运入口；
`python tools/plot_field_study.py` 只计算静态场，导出 `docs/figures/field_scale_static/` 中的独立 PNG/SVG。
不要在“只检查”阶段运行整个 `pytest` 或 `tools/execute_notebooks.py`：旧测试/Notebook 中含真实输运调用。

### 35 组具体是什么

所有正式组 `n_simulated=10000000`，主曝光 `exposure_electrons=1000000`。
R=50 μm、T=298 K、ε水=78、表面张力 0.072 N/m、3 MeV、L1=10 mm。
源的相对 RMS 能散=0.001，x–x′/y–y′ 相关系数=0，单轴 RMS 束散=10 mrad。
探测器 PSF=0、像素=25 μm、直径/视野=40 mm；效率=1、平均增益=1、Gamma shape=4、背景/读出噪声=0，均为情景参数。

| 组别 | 条件 | 数量 |
|---|---|---:|
| 主组 | 无滴、无场水滴、1 V/0.5 nm 偶极层、PB 0.01/0.1/1 μM、表面净电荷 ±10⁶e；各用 10/5/1 μm 焦点，L2=2000 mm | 24 |
| 模型补充 | 10 μm 焦点、L2=2000 mm；±10⁵e、±10⁷e、1 V/1 nm 偶极层、±10⁸e/10 nm 中性双壳 | 6 |
| 短距离 | 10 μm 焦点、L2=500 mm；无滴、无场水滴、0.5 nm 偶极层、PB 0.1 μM、+10⁶e | 5 |

完整配置在 `examples/field_scale_study/`，元数据在 `manifest.json`。
`data/field_scale_sources.json` 区分原文数值、假设、单位换算和计算值，并给出 DOI、PDF 文件名和页码。
运行无需原始 PDF。固定束散缩焦同时降低发射度，不能解释成同一束流只调透镜。
8 MV/cm 的局域探针场不直接换算为净电荷，也不作为独立场重复叠加到双电层。

## 3. Linux 上显式执行数值验证

**以下命令从这里开始会发射电子。此步骤本次交付未执行。**

```bash
python tools/validate_field_study.py --run --output results/field_scale_validation
python -m pytest -q tests/test_native.py tests/test_spherical_pipeline.py
python tools/validate_transport.py --n 3000 --output results/water_transport_validation
```

第一条使用理想点源、无材料的确定性射线，扫描几何入射半径并在界面两侧加密。
每组比较原步长与全部相关步长减半后的落点，输出 `probes.npz`、`validation.json`、PNG/SVG。
检查静电能量守恒与落点变化小于 0.01 像素；位移大于 1 μm 的探针另检查相对变化小于 1%。
几何入射半径不是电场作用后的实际最近距离。只有真空数值检查通过，不能证明水中散射也已收敛。

后两条执行已有轨迹/材料回归与水中散射、能损、生产截断敏感性检查。
水材料结果和有限样本波动仍需人工检查。不要把统计检验的 p 值大解释成两个设置完全等价。
验证不通过时保留日志、收紧参数再检查，不通过降低精度绕过。
探针目录拒绝覆盖；重跑时换一个输出目录。

## 4. 正式运行、续跑和资源

先运行一个条件确认目标机速度及空间，再运行其余组：

```bash
python tools/run_field_scale_study.py --run --case neutral_s10_l2000
python tools/run_field_scale_study.py --run --all --resume
```

指定几个条件时可重复 `--case`。不使用默认目录时，后续所有命令都传同一个 `--output`：

```bash
python tools/run_field_scale_study.py --run \
  --case absent_s10_l2000 --case neutral_s10_l2000 \
  --case dipole_0p5nm_s10_l2000 --output results/my_study
python tools/run_field_scale_study.py --run --all --resume --output results/my_study
```

条件之间串行，每组自动留一逻辑核、最多 8 个 worker；可在各 YAML 的 `geant4_threads` 显式指定。
改变配置后不要继续复用旧输出根目录。跨平台、线程数或版本不承诺逐事件随机轨迹相同。
总量 3.5 亿初级事件，不是 3.5 亿探测器落点；探测器孔径和水中散射会降低接收比例。

脚本给出的内存约 6.6 GB/组、磁盘约 182 GB/全研究，是按事件数的粗略估计而非实测或上限。
次级粒子、CSV 和压缩率会改变占用；Linux 上应预留额外空间，建议至少 16 GB 内存并监测实际峰值。
正式运行前每组检查估计可用磁盘空间，不足则在启动输运前报错，不减少电子数。
首组的 `result.json` 给出目标机实际耗时；不要把本机以外的旧速度当作 Linux 的实测值。

每组目录包含 `state.json` 及独立 `attempt-0001/`。成功前验证：

1. 配置快照与运行参数一致，代码/二进制未在运行期间改变；
2. 原生 worker 完成的事件总数正确，落点/沉积数组及图像结构有效；
3. 保存全部必要文件的 SHA-256，再写 `complete`。

`--resume` 只复用配置、版本指纹、文件校验均匹配的成功结果。失败尝试保留原始数据和错误状态，
重试写到新的 attempt；损坏的已完成结果或版本不匹配会报错，不悄悄覆盖。
同一输出目录有调度锁；进程被强行结束后可能残留 `.study.lock`，须先人工确认无进程使用该目录，才能移走此锁文件。
不要让两个计算任务共用输出目录。

## 5. 只读取已有数据进行后处理

```bash
python tools/run_field_scale_study.py --analyze --all
jupyter lab notebooks/06_field_scale_comparison.ipynb
```

也可 `--analyze --case dipole_0p5nm_s10_l2000`，会同时读取匹配的无场水滴/无滴参照。
缺少参照时只报告缺失，不补跑输运。`--analyze` 会完整检查输入哈希；对大文件需要一定读取时间。
Notebook 默认只读已生成分析文件，并展示静态场与几何分辨率；没有结果会打印“尚未运行”。
打开 Notebook 不等于重新做完整文件哈希校验，页面会明确此限制。

每个完成条件的 `analysis/条件名/` 保存：

- `exposure_100000/1000000/10000000.npz` 及 PNG/SVG：相同原始落点分别生成曝光，绝不在已有 observed 上再加噪；
- `profiles.npz/csv`：原始落点的径向环计数、分类密度及十个初级事件子样本；
- `classified_profiles`：未穿滴初级、穿滴初级、次级及总量，分母包括无落点初级；
- `comparison`：全图、固定物方边缘局部图及未经额外平滑的差分；
- `radial_comparison`：按一个探测器像素宽度分箱的图像径向平均；
- `analysis.json`：输入哈希、配置、后处理版本、曝光信号与束流电荷交换估计。

三种焦点齐全时生成 `analysis/focus/模型.png/svg` 全图、`模型_edge.png/svg` 边缘局部，
以及 `模型_radial.png/svg` 三焦点绝对径向强度/差分叠加。共同色标，绝对通量不分别归一化。
分类曲线按真实落点半径统计，单位是 hits / incident primary / detector mm²；
图像径向曲线按像素中心分箱，单位是平均信号/像素，两者不要直接相减。
不同 L2 的图像放大率不同，不能未经物方坐标换算直接逐像素相减。

期望图仍有有限 Monte Carlo 采样纹理；十个子样本曲线只展示波动，不是置信区间。
关闭 PSF 不消除源尺寸、水中散射、像素积分。Q=0 时 |ΔQ|/|Q| 不可定义，保存为 null。
没有自动“可见/不可见”结论，不运行电荷拟合或最低检出限判定。

## 6. 证据、已完成部分与后续路径

交付证据与验收记录见 [开发交付记录](field_scale_delivery.md)。
配置清单证明 35 组的定义，不证明它们已运行；轻量测试证明程序分支和数据处理行为，不证明 Geant4 物理正确。
路径是：**源码/配置检查 → Linux 编译与数值/材料验证 → 正式输运 → 后处理 → 人工阅读**。
每一步的输出与下一步入口分离，缺失结果不替换为零、假数据或成功标记。

需要再次生成源码迁移包时：

```bash
python tools/package_linux_source.py --output artifacts/droplet-shadow-linux-source.tar.gz
```

打包当前工作区而非 Git HEAD，所以包含尚未提交但需要的修改；不提交、不推送。
排除环境、二进制、旧结果和缓存。旧 Notebook 的历史输出只在包内清空，原文件不改；
06 的静态场/检查输出保留。包内 `TRANSFER_MANIFEST.json` 记录归档文件哈希和这种转换。
