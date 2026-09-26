"""打包当前工作区的可迁移源码，不包含环境、二进制、旧结果和缓存。

不同于 git archive，本工具包含未提交/未跟踪但在白名单源码目录中的文件。
旧 Notebook 的历史执行输出仅在归档副本中移除，绝不改动原工作区文件。
归档清单记录实际打包内容的哈希，解包后可独立核对。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src", "cpp", "tests", "tools", "examples", "data", "docs", "notebooks")
ROOT_FILES = ("README.md", "pyproject.toml", "environment-linux.yml", ".gitignore", "LICENSE", "LICENSE.md")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store", "build", "results"}
EXTENSIONS = {".py", ".cpp", ".h", ".hpp", ".txt", ".md", ".yaml", ".yml", ".json", ".csv", ".ipynb", ".png", ".svg"}


def source_files(root: Path):
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for folder in DIRECTORIES:
        for path in sorted((root / folder).rglob("*")):
            if (path.is_file() and not path.is_symlink() and path.suffix in EXTENSIONS and
                    not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.relative_to(root).parts)):
                files.append(path)
    return sorted(set(files))


def payload(path: Path):
    original = path.read_bytes()
    transformed = False
    data = original
    if path.suffix == ".ipynb" and path.name != "06_field_scale_comparison.ipynb":
        notebook = json.loads(original)
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        data = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode()
        transformed = True
    return data, hashlib.sha256(original).hexdigest(), transformed


def create_package(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Archive exists, choose a new filename: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(output, "w:gz") as archive:
        def add(name, data):
            info = tarfile.TarInfo("droplet-shadow-imaging/" + name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
        for path in source_files(root):
            data, original_hash, transformed = payload(path)
            name = path.relative_to(root).as_posix()
            add(name, data)
            entries.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                            "source_sha256": original_hash, "legacy_notebook_outputs_removed": transformed})
        manifest = {"schema": 1, "scope": "Working-tree source, not Git HEAD; no environments/binaries/results",
                    "formal_transport_performed_for_delivery": False, "files": entries}
        add("TRANSFER_MANIFEST.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    # 读取归档内容逐文件校验，避免仅凭 tar 命令成功就声称包完整。
    with tarfile.open(output, "r:gz") as archive:
        for entry in entries:
            content = archive.extractfile("droplet-shadow-imaging/" + entry["path"]).read()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise RuntimeError("Archive verification failed: " + entry["path"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/droplet-shadow-linux-source.tar.gz")
    args = parser.parse_args()
    manifest = create_package(ROOT, args.output)
    print(f"Verified {len(manifest['files'])} source files in {args.output.resolve()}")


if __name__ == "__main__":
    main()
