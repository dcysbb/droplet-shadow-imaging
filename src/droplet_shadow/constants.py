"""全项目共用的 SI 常数。

Python 电场、轨迹和统计计算统一使用 m、kg、s、C、J、V；只有
``native.py`` 导出 CSV 时才转换为 Geant4 约定的 mm、MeV 等单位。
这样可以避免同一个公式在不同模块中暗中混用 μm、mm 和 m。
"""

import math

# 真空介电常数 F/m；E_CHARGE 是正的元电荷，电子电荷为 -E_CHARGE。
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837139e-31  # kg，电子静质量
C = 299792458.0  # m/s
K_B = 1.380649e-23  # J/K
TWO_PI = 2.0 * math.pi
COULOMB_K = 1.0 / (4.0 * math.pi * EPS0)  # φ=kQ/r 中的 k
# 高斯分布的 FWHM = FWHM_SIGMA × 标准差 σ。
FWHM_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
