"""单液滴电子点投影成像模拟的最小 Python 公共接口。

通常从这里导入 ``load_config``、``build_field``、``simulate``；
需要检查离子、图像拟合或 Geant4 CSV 时再进入相应子模块。
"""

from .config import SimulationConfig, load_config
from .fields import build_field
from .simulation import simulate
from .sensitivity import detection_calibration

__all__ = ["SimulationConfig", "load_config", "build_field", "simulate", "detection_calibration"]
