"""只验证新的 06 Notebook；显式 --execute 才执行静态场并保存输出。

执行时临时插入禁止电子输运的保护单元。保护单元不会保存到 Notebook，
也不会影响用户随后在 Linux 上使用其他工具；原有 01–05 完全不执行。
"""

import argparse
import base64
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/06_field_scale_comparison.ipynb"
GUARD = """
import sys
sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'src'))
import droplet_shadow as _package
import droplet_shadow.simulation as _simulation
import droplet_shadow.native as _native
import droplet_shadow.rays as _rays
def _forbid_transport(*args, **kwargs):
    raise RuntimeError('Electron transport is prohibited during static notebook validation')
_package.simulate = _forbid_transport
_simulation.simulate = _forbid_transport
_simulation.run_geant4 = _forbid_transport
_simulation.trace_rays = _forbid_transport
_native.run_geant4 = _forbid_transport
_native._run_native_process = _forbid_transport
_rays.trace_rays = _forbid_transport
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    if not args.execute:
        print("06 Notebook schema valid. No cells executed; use --execute for guarded static evaluation.")
        return
    notebook.cells.insert(0, nbformat.v4.new_code_cell(GUARD))
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    notebook.cells.pop(0)
    # 删除临时单元后，显示的执行序号也从 1 开始。
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.execution_count is not None:
            cell.execution_count -= 1
        for output in cell.get("outputs", []):
            if output.get("execution_count") is not None:
                output.execution_count -= 1
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    # 从已执行输出导出原图用于人工视觉检查；不是新的模拟结果。
    destination = ROOT / "artifacts/field_scale_notebook_review"
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                count += 1
                (destination / f"static-field-{count:02d}.png").write_bytes(base64.b64decode(encoded))
    print(f"Executed only notebook 06 with transport blocked; exported {count} static figures to {destination}")


if __name__ == "__main__":
    main()
