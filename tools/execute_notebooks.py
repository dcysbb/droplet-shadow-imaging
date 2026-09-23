"""Execute, save and validate notebooks with the active Python kernel."""

from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    executed = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                              resources={"metadata": {"path": str(root)}}).execute()
    nbformat.validate(executed)
    nbformat.write(executed, path)
    print(f"Executed {path.name}")

