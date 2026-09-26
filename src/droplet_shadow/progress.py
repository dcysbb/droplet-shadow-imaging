"""轻量运行进度：终端原地刷新，Notebook 更新同一输出，重定向时限频。

不依赖 tqdm/ipywidgets。百分比只描述完整初级事件的输运进度，
不是探测器命中数，也不是实验曝光数；100% 后可能仍在合并或保存。
"""

from __future__ import annotations

from html import escape
import math
import sys
import time


class SimulationProgress:
    """由 Python 主线程显示；Geant4 worker 只提供累计计数，不直接打印。"""

    def __init__(self, total: int, enabled: bool = True, stream=None):
        self.total, self.enabled = total, enabled
        self.stream = sys.stderr if stream is None else stream
        self.done = 0
        self.phase = "准备"
        self.started = None
        self.last_render = -float("inf")
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.display = None
        self.notebook = False
        if enabled and stream is None and "ipykernel" in sys.modules:
            try:
                from IPython import get_ipython
                self.notebook = getattr(get_ipython(), "kernel", None) is not None
            except ImportError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stage("完成" if exc_type is None else "已中断" if exc_type is KeyboardInterrupt else "失败")
        if self.enabled and self.tty:
            self.stream.write("\n")
            self.stream.flush()
        return False

    def stage(self, name: str):
        if not self.enabled or name == self.phase:
            return
        self.phase = name
        if name == "Geant4 输运":
            self.started = time.monotonic()
        self.render(force=True)

    def update(self, completed: int):
        if not 0 <= completed <= self.total:
            raise ValueError("Progress count outside event range")
        self.done = max(self.done, completed)  # 无序/重复的进度消息不能令进度倒退。
        self.render()

    def render(self, force=False):
        if not self.enabled:
            return
        now = time.monotonic()
        interval = .2 if self.tty or self.notebook else 5.0
        if not force and now - self.last_render < interval:
            return
        self.last_render = now
        fraction = self.done / self.total if self.total else 0
        filled = int(24 * fraction)
        text = (f"{self.phase} [{'#' * filled}{'-' * (24-filled)}] "
                f"{fraction:6.1%}  {self.done:,}/{self.total:,} 初级电子")
        if self.started is not None:
            elapsed = max(now - self.started, 0)
            rate = self.done / elapsed if elapsed else 0
            # 刚启动时样本过少，先不猜 ETA；向上取整避免未完成却显示剩余 0 秒。
            eta = f"{math.ceil((self.total-self.done)/rate)}s" if rate and elapsed >= 1 else "--"
            if self.phase == "Geant4 输运":
                text += f" | {rate:,.0f} e-/s | 已用 {elapsed:.0f}s | 预计剩余 {eta}"
        try:
            if self.notebook:
                from IPython.display import HTML, display
                content = HTML(f'<progress value="{self.done}" max="{self.total}" '
                               f'style="width:240px"></progress><pre>{escape(text)}</pre>')
                if self.display is None:
                    self.display = display(content, display_id=True)
                else:
                    self.display.update(content)
            elif self.tty:
                self.stream.write("\r\033[2K" + text)  # 清整行，避免中文宽度造成残影。
                self.stream.flush()
            else:
                self.stream.write(text + "\n")
                self.stream.flush()
        except (OSError, ImportError):
            # 进度显示不能因管道关闭或可选 Notebook 前端缺失而破坏模拟。
            self.enabled = False
