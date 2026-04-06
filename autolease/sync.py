"""Code file sync to cluster via rsync."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from .config import PoolConfig

# Default: only sync code files
DEFAULT_INCLUDE = [
    "*.py", "*.sh", "*.bash", "*.fish",
    "*.yaml", "*.yml", "*.json", "*.toml", "*.cfg", "*.ini",
    "*.txt", "*.md", "*.rst",
    "Makefile", "Dockerfile", "*.dockerfile",
    "requirements*.txt", "pyproject.toml", "setup.py", "setup.cfg",
    ".env.example",
]

DEFAULT_EXCLUDE = [
    "__pycache__/",
    "*.pyc",
    ".git/",
    "*.egg-info/",
    ".eggs/",
    "node_modules/",
    ".venv/", "venv/",
    "wandb/",
    ".ipynb_checkpoints/",
]

def _detect_project_root() -> Path:
    """Find git root or cwd."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except Exception:
        pass
    return Path.cwd()


def _relative_to_home(path: Path) -> str:
    """Get path relative to user home for remote mirroring."""
    home = Path.home()
    try:
        return str(path.relative_to(home))
    except ValueError:
        return f"projects/{path.name}"


def sync(config: PoolConfig,
         local_dir: Optional[str] = None,
         dry_run: bool = False,
         include: Optional[list[str]] = None,
         exclude: Optional[list[str]] = None,
         verbose: bool = False) -> subprocess.CompletedProcess:
    """Rsync code files from local project to cluster.
    Remote path mirrors local path relative to ~."""
    project_root = Path(local_dir) if local_dir else _detect_project_root()
    rel_path = _relative_to_home(project_root)
    remote_dir = f"~/{rel_path}/"

    inc = include or DEFAULT_INCLUDE
    exc = exclude or DEFAULT_EXCLUDE

    cmd = [
        "rsync", "-az", "--update",  # skip files newer on remote
    ]

    # Excludes first — these take priority
    for pattern in exc:
        cmd.append(f"--exclude={pattern}")

    # Then include dirs (for traversal) + code file patterns
    cmd.append("--include=*/")
    for pattern in inc:
        cmd.append(f"--include={pattern}")

    # Exclude everything else
    cmd.append("--exclude=*")

    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.append("-v")

    # Source must end with / to sync contents
    src = str(project_root).rstrip("/") + "/"
    dst = f"{config.ssh_host}:{remote_dir}"

    cmd.extend([src, dst])

    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def get_remote_dir(config: PoolConfig, local_dir: Optional[str] = None) -> str:
    """Get the remote sync directory for a local project."""
    project_root = Path(local_dir) if local_dir else _detect_project_root()
    rel_path = _relative_to_home(project_root)
    return f"~/{rel_path}"


def pull(config: PoolConfig,
         remote_subpath: str,
         local_dest: Optional[str] = None,
         verbose: bool = False) -> subprocess.CompletedProcess:
    """Pull files from the remote project dir to local."""
    project_root = Path(local_dest) if local_dest else _detect_project_root()
    rel_path = _relative_to_home(project_root)
    remote_dir = f"~/{rel_path}"

    src = f"{config.ssh_host}:{remote_dir}/{remote_subpath}"
    dst = str(project_root / remote_subpath)

    cmd = ["rsync", "-az", src, dst]
    if verbose:
        cmd.append("-v")

    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)
