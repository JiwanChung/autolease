"""Configuration management for autolease."""

import json
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
    # Tunables (defaults match historical hardcoded values)
    tui_refresh_interval: float = 30.0  # TUI background refresh, seconds
    poll_interval: float = 10.0         # `autolease poll` cycle, seconds
    poll_tail_lines: int = 30           # initial lines on each poll cycle
    ssh_timeout: int = 10               # per-SSH-call timeout, seconds
    mtime_threshold: int = 60           # output-file recency for liveness, seconds
    discovery_cache_seconds: int = 300  # how long to cache scontrol/sinfo output

    def __post_init__(self):
        if not self.state_dir:
            self.state_dir = str(_data_dir())

    @property
    def state_path(self) -> str:
        return os.path.expanduser(self.state_dir)


def _discovery_cache_path() -> Path:
    return _data_dir() / "discovery.cache.json"


def _load_discovery_cache(max_age_seconds: int) -> Optional[dict]:
    """Read PARTITION_INFO from cache if fresh enough."""
    p = _discovery_cache_path()
    if not p.exists():
        return None
    try:
        import time as _time
        age = _time.time() - p.stat().st_mtime
        if age > max_age_seconds:
            return None
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _save_discovery_cache(partitions: dict) -> None:
    """Persist PARTITION_INFO so subsequent CLI invocations skip the
    scontrol/sinfo round-trips."""
    p = _discovery_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = str(p) + ".tmp"
        # Convert tuple values to lists for JSON serialization
        data = {name: [list(qos), gpu] for name, (qos, gpu) in partitions.items()}
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, p)
    except Exception:
        pass


def discover_partitions(slurm, max_age_seconds: int = 300, force: bool = False) -> None:
    """Populate PARTITION_INFO from the live cluster via scontrol.
    Caches the result for `max_age_seconds` to avoid hammering the cluster
    on every CLI invocation. Pass force=True to skip the cache."""
    if not force:
        cached = _load_discovery_cache(max_age_seconds)
        if cached is not None:
            PARTITION_INFO.clear()
            for name, (qos, gpu) in cached.items():
                PARTITION_INFO[name] = (list(qos), gpu)
            return

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

    # Collect all QoS names mentioned across partitions
    all_qos_names: set[str] = set()
    parsed_partitions: list[tuple[str, str, str]] = []  # (name, allow_qos, gpu)
    for line in r.stdout.strip().splitlines():
        info = {}
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                info[k] = v
        name = info.get("PartitionName", "")
        allow_qos = info.get("AllowQos", "")
        if not name:
            continue
        gpu = gpu_types.get(name, "unknown")
        if allow_qos and allow_qos != "ALL":
            qos_list = [q.strip() for q in allow_qos.split(",") if q.strip()]
            all_qos_names.update(qos_list)
        parsed_partitions.append((name, allow_qos, gpu))

    for name, allow_qos, gpu in parsed_partitions:
        if not allow_qos:
            continue
        if allow_qos == "ALL":
            # Use all known QoS names (from other partitions + config)
            known = all_qos_names | set(QOS_GPU_LIMITS.keys())
            # Prefer limited QoS first (guaranteed slots), unlimited last
            qos_list = sorted(known,
                              key=lambda q: (0 if QOS_GPU_LIMITS.get(q, 0) > 0 else 1, q))
            if not qos_list:
                qos_list = ["base_qos"]
        else:
            qos_list = [q.strip() for q in allow_qos.split(",") if q.strip()]
        PARTITION_INFO[name] = (qos_list, gpu)

    # Persist for subsequent invocations
    _save_discovery_cache(PARTITION_INFO)


def apply_qos_config(cfg: PoolConfig) -> None:
    """Apply QoS limits from config to the module-level dict."""
    QOS_GPU_LIMITS.update({
        name: r.gpu_limit for name, r in cfg.qos_rules.items()
    })


def pick_qos(partition: str, num_gpus: int, usage: dict[str, int]) -> str:
    """Auto-select the best QoS for a partition given current usage."""
    info = PARTITION_INFO.get(partition)
    if info:
        preference = info[0]
    else:
        # Partition not in PARTITION_INFO (AllowQos empty or discovery failed).
        # Build preference from all configured QoS, limited first.
        preference = sorted(
            QOS_GPU_LIMITS.keys(),
            key=lambda q: (0 if QOS_GPU_LIMITS.get(q, 0) > 0 else 1, q),
        )
        if not preference:
            return "base_qos"

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
        cfg = PoolConfig()
    else:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        qos_rules = {}
        for name, rule in raw.get("qos", {}).items():
            if isinstance(rule, dict):
                qos_rules[name] = QoSRule(name=name, gpu_limit=int(rule.get("gpu_limit", 0)))
            else:
                qos_rules[name] = QoSRule(name=name, gpu_limit=int(rule))

        # Tunables block (all optional, defaults from PoolConfig)
        tun = raw.get("tunables", {}) or {}
        cfg = PoolConfig(
            ssh_host=raw.get("ssh_host", "localhost"),
            shell=raw.get("shell", "bash"),
            env=raw.get("env", ""),
            env_activate=raw.get("env_activate", "micromamba run -n {env}"),
            exclude_nodes=raw.get("exclude_nodes", []),
            state_dir=raw.get("state_dir", str(_data_dir())),
            qos_rules=qos_rules,
            tui_refresh_interval=float(tun.get("tui_refresh_interval", 30.0)),
            poll_interval=float(tun.get("poll_interval", 10.0)),
            poll_tail_lines=int(tun.get("poll_tail_lines", 30)),
            ssh_timeout=int(tun.get("ssh_timeout", 10)),
            mtime_threshold=int(tun.get("mtime_threshold", 60)),
            discovery_cache_seconds=int(tun.get("discovery_cache_seconds", 300)),
        )

    # Environment variable overrides (take precedence over config file)
    if os.environ.get("AUTOLEASE_SSH_HOST"):
        cfg.ssh_host = os.environ["AUTOLEASE_SSH_HOST"]
    if os.environ.get("AUTOLEASE_SHELL"):
        cfg.shell = os.environ["AUTOLEASE_SHELL"]
    if os.environ.get("AUTOLEASE_ENV"):
        cfg.env = os.environ["AUTOLEASE_ENV"]

    return cfg
