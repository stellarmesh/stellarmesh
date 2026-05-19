"""Bump version in pyproject.toml and pixi.toml, then create a git tag."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def bump(version: str) -> None:
    """Rewrite the version literal in both manifests, then commit and tag."""
    for path in [ROOT / "pyproject.toml", ROOT / "pixi.toml"]:
        content = path.read_text()
        updated = re.sub(
            r'^version = ".*"', f'version = "{version}"', content, flags=re.MULTILINE
        )
        if updated == content:
            print(f"Warning: no version line found in {path.name}")
        path.write_text(updated)

    subprocess.run(["git", "add", "pyproject.toml", "pixi.toml"], check=True, cwd=ROOT)
    subprocess.run(
        ["git", "commit", "-m", f"chore: bump version to {version}"],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(["git", "tag", f"v{version}"], check=True, cwd=ROOT)
    print(f"Bumped to {version} and created tag v{version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: pixi run bump <version>")
        sys.exit(1)
    bump(sys.argv[1])
