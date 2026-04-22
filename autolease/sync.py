"""Code file sync to cluster via rsync."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from .config import PoolConfig


def _newest_mtime(root: Path, include: list[str]) -> float:
    """Find the newest mtime among code files in a directory."""
    import fnmatch
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded dirs
        dirnames[:] = [d for d in dirnames if d not in (
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            "wandb", ".eggs", ".ipynb_checkpoints",
        ) and not d.endswith(".egg-info")]
        for f in filenames:
            if any(fnmatch.fnmatch(f, pat) for pat in include):
                mt = os.path.getmtime(os.path.join(dirpath, f))
                if mt > newest:
                    newest = mt
    return newest

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
    ".autolease_sync",
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


def needs_sync(local_dir: Optional[str] = None,
               include: Optional[list[str]] = None) -> bool:
    """Check if any code file has been modified since last sync."""
    project_root = Path(local_dir) if local_dir else _detect_project_root()
    stamp_file = project_root / ".autolease_sync"
    inc = include or DEFAULT_INCLUDE

    newest = _newest_mtime(project_root, inc)
    if newest == 0.0:
        return False  # no code files at all

    if stamp_file.exists():
        last_sync = stamp_file.stat().st_mtime
        if newest <= last_sync:
            return False  # nothing changed since last sync

    return True


def _touch_stamp(local_dir: Optional[str] = None):
    """Update the sync timestamp."""
    project_root = Path(local_dir) if local_dir else _detect_project_root()
    stamp_file = project_root / ".autolease_sync"
    stamp_file.touch()


def sync(config: PoolConfig,
         local_dir: Optional[str] = None,
         dry_run: bool = False,
         include: Optional[list[str]] = None,
         exclude: Optional[list[str]] = None,
         verbose: bool = False,
         force: bool = False) -> Optional[subprocess.CompletedProcess]:
    """Rsync code files from local project to cluster.
    Skips if no code files changed since last sync (unless force=True).
    Returns None if skipped, CompletedProcess otherwise."""
    if not force and not dry_run and not needs_sync(local_dir, include):
        return None  # nothing changed

    project_root = Path(local_dir) if local_dir else _detect_project_root()
    rel_path = _relative_to_home(project_root)
    remote_dir = f"~/{rel_path}/"

    inc = include or DEFAULT_INCLUDE
    exc = exclude or DEFAULT_EXCLUDE

    # Let rsync reuse the autolease SSH ControlMaster connection
    from .slurm import _control_socket_path
    ssh_cmd = (
        "ssh -o BatchMode=yes -o ConnectTimeout=10"
        " -o ControlMaster=auto"
        f" -o ControlPath={_control_socket_path()}"
        " -o ControlPersist=10m"
    )
    cmd = [
        "rsync", "-az", "--update",  # skip files newer on remote
        "-e", ssh_cmd,
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0 and not dry_run:
        _touch_stamp(local_dir)
    return result


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

    from .slurm import _control_socket_path
    ssh_cmd = (
        "ssh -o BatchMode=yes -o ConnectTimeout=10"
        " -o ControlMaster=auto"
        f" -o ControlPath={_control_socket_path()}"
        " -o ControlPersist=10m"
    )
    cmd = ["rsync", "-az", "-e", ssh_cmd, src, dst]
    if verbose:
        cmd.append("-v")

    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)
