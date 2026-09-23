"""从上到下重跑四个 Notebook，并把新的执行结果写回原 .ipynb。

这是可复现性检查，不会替代正式高统计量扫描。运行前应确认
``python3`` Jupyter kernel 指向已安装 droplet_shadow/Geant4 的环境；
脚本固定把工作目录设为项目根目录，使 Notebook 中的相对路径稳定。
注意：本脚本会更新受 Git 跟踪的 Notebook 输出和执行计数。
"""

from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "notebooks").glob("*.ipynb")):
    # 文件名 01..04 规定阅读/执行顺序；单本内部按原单元格顺序运行。
    notebook = nbformat.read(path, as_version=4)
    executed = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                              resources={"metadata": {"path": str(root)}}).execute()
    nbformat.validate(executed)
    nbformat.write(executed, path)
    print(f"Executed {path.name}")
