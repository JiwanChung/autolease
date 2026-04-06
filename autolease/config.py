"""Configuration management for autolease."""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Well-known GPU VRAM (GB) — no config needed
GPU_VRAM = {
    "RTX3090": 24, "A5000": 24, "RTX4090": 24,
    "A6000": 48, "RTX6000ADA": 48, "RTXPRO6000": 48,
    "a100": 80, "A100": 80, "H100": 80,
}

# Populated at runtime from Slurm (scontrol show partition)
PARTITION_INFO: dict[str, tuple[list[str], str]] = {}

# Populated from config
QOS_GPU_LIMITS: dict[str, int] = {}


def _config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "autolease"
    return Path.home() / ".config" / "autolease"


def _data_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "autolease"
    return Path.home() / ".local" / "share" / "autolease"


@dataclass
class QoSRule:
    name: str
    gpu_limit: int = 0  # 0 = unlimited
    prefer_over: Optional[str] = None  # fallback QoS when this one is full


@dataclass
class LeaseSpec:
    partition: str
    qos: str
    num_gpus: int = 1
    time: Optional[str] = None
    exclude: str = ""

    @property
    def gpu_type(self) -> str:
        info = PARTITION_INFO.get(self.partition)
        return info[1] if info else "unknown"

    @property
    def vram_gb(self) -> int:
        return GPU_VRAM.get(self.gpu_type, 0)


@dataclass
class PoolConfig:
    ssh_host: str = "localhost"
    shell: str = "bash"  # remote shell for job execution (bash, fish, zsh)
    env: str = ""  # default conda/micromamba env for jobs
    env_activate: str = "micromamba run -n {env}"  # command template, {env} replaced
    exclude_nodes: list[str] = field(default_factory=list)
    state_dir: str = ""
    qos_rules: dict[str, QoSRule] = field(default_factory=dict)

    def __post_init__(self):
        if not self.state_dir:
            self.state_dir = str(_data_dir())

    @property
    def state_path(self) -> str:
        return os.path.expanduser(self.state_dir)


def discover_partitions(slurm) -> None:
    """Populate PARTITION_INFO from the live cluster via scontrol."""
    from .slurm import Slurm
    try:
        r = slurm.cfg.run(
            'scontrol show partition --oneliner',
            timeout=15,
        )
        if r.returncode != 0:
            return
    except Exception:
        return

    # Also get GPU types per partition from sinfo
    gpu_types: dict[str, str] = {}
    try:
        r2 = slurm.cfg.run('sinfo -o "%P|%G" --noheader', timeout=10)
        if r2.returncode == 0:
            for line in r2.stdout.strip().splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    part = parts[0].rstrip("*")
                    gres = parts[1]
                    if gres and gres != "(null)" and ":" in gres:
                        gparts = gres.split(":")
                        if len(gparts) >= 2:
                            gpu_types.setdefault(part, gparts[1])
    except Exception:
        pass

    PARTITION_INFO.clear()
    for line in r.stdout.strip().splitlines():
        info = {}
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                info[k] = v
        name = info.get("PartitionName", "")
        allow_qos = info.get("AllowQos", "")
        if not name or allow_qos in ("", "ALL"):
            continue
        qos_list = [q.strip() for q in allow_qos.split(",") if q.strip()]
        gpu = gpu_types.get(name, "unknown")
        PARTITION_INFO[name] = (qos_list, gpu)

    QOS_GPU_LIMITS.clear()


def apply_qos_config(cfg: PoolConfig) -> None:
    """Apply QoS limits from config to the module-level dict."""
    QOS_GPU_LIMITS.update({
        name: r.gpu_limit for name, r in cfg.qos_rules.items()
    })


def pick_qos(partition: str, num_gpus: int, usage: dict[str, int]) -> str:
    """Auto-select the best QoS for a partition given current usage."""
    info = PARTITION_INFO.get(partition)
    if not info:
        return "base_qos"
    preference = info[0]

    for qos in preference:
        limit = QOS_GPU_LIMITS.get(qos, 0)
        current = usage.get(qos, 0)
        if limit == 0 or current + num_gpus <= limit:
            return qos
    return preference[-1]


def config_path() -> Path:
    return _config_dir() / "config.yaml"


def load_config(path: Optional[str] = None) -> PoolConfig:
    if path is None:
        p = config_path()
        if p.exists():
            path = str(p)
    if path is None:
        return PoolConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    qos_rules = {}
    for name, rule in raw.get("qos", {}).items():
        if isinstance(rule, dict):
            qos_rules[name] = QoSRule(name=name, gpu_limit=int(rule.get("gpu_limit", 0)))
        else:
            qos_rules[name] = QoSRule(name=name, gpu_limit=int(rule))

    return PoolConfig(
        ssh_host=raw.get("ssh_host", "localhost"),
        shell=raw.get("shell", "bash"),
        env=raw.get("env", ""),
        env_activate=raw.get("env_activate", "micromamba run -n {env}"),
        exclude_nodes=raw.get("exclude_nodes", []),
        state_dir=raw.get("state_dir", str(_data_dir())),
        qos_rules=qos_rules,
    )
