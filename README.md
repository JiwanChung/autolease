# autolease

Personal GPU pool manager for shared Slurm clusters.

autolease holds GPU allocations as persistent Slurm jobs and lets you (or your coding agents) submit work to them without fighting for resources each time. It handles QoS selection, bad-node detection, lease renewal, job queuing, code sync, and conda env activation.

## Why

On a competitive shared GPU cluster:

- **Allocations are slow to get.** You don't want to `salloc` for every experiment.
- **Nodes are unreliable.** CUDA versions, driver states, and hardware health vary across nodes.
- **QoS rules are complex.** Guaranteed slots are limited; preemptible slots are unlimited but risky.
- **Multiple projects compete.** You run 8-10 projects and they shouldn't block each other.

autolease solves this by maintaining a personal pool of held GPU allocations and routing work to them.

## How it works

1. **Leases** are persistent `sbatch sleep` jobs that hold GPUs. You acquire them with `autolease up <partition>`.
2. **Jobs** are async. `autolease run` syncs your code, wraps it in your conda env, dispatches to a lease, and returns a job ID.
3. **Priority & preemption**: higher-priority jobs (`-P 10`) preempt lower-priority running jobs. Preempted jobs are re-queued, not killed. Jobs prefer the smallest GPU that fits.
4. **Code sync**: code files are auto-rsynced to the cluster before each job. Remote path mirrors your local path relative to `~`.
5. **Health checks** run `nvidia-smi` on each lease. Bad nodes are auto-excluded.
6. **QoS auto-selection** fills guaranteed slots first, then overflows to preemptible.
7. **Preemption detection** notifies you when a lease disappears unexpectedly.

## Install

```bash
# With uv (recommended)
uv tool install git+https://github.com/JiwanChung/autolease

# Or from a local clone
git clone https://github.com/JiwanChung/autolease
cd autolease
uv tool install -e .
```

Requires Python 3.9+ and SSH access to the cluster. The cluster needs Slurm 23.11+.

## Quick start

```bash
# Launch the interactive TUI
autolease

# Or use the CLI:
autolease up suma_rtx4090              # acquire 1 RTX4090, auto QoS, max time
autolease up suma_a6000 -n 4           # acquire 4 A6000s
autolease pool                         # see your leases

# Submit work (auto-syncs code, activates env)
export AUTOLEASE_JOB_ID=$(autolease run -- python train.py)

# Per-shell job shortcuts (reads AUTOLEASE_JOB_ID):
autolease poll                         # tail stdout, refreshing every 10s
autolease log                          # read stdout
autolease log --stderr                 # read stderr
autolease status                       # queued / running / done:0 / failed
autolease cancel                       # kill the job
autolease redo                         # re-submit the same command

# Or pass job IDs explicitly:
autolease log 42
autolease cancel 42

# Submit and immediately start polling:
autolease run --poll -- python train.py

# Done for the day
autolease down                         # release all leases
```

## Configuration

Config lives at `~/.config/autolease/config.yaml`:

```yaml
ssh_host: your-cluster
shell: fish              # remote shell (bash, fish, zsh)

# Default conda/micromamba env for jobs
env: dl
env_activate: "micromamba run -n {env}"   # or "conda run -n {env}"

exclude_nodes:
  - bad-node-01

# QoS GPU limits per user (0 or omitted = unlimited)
qos:
  base_qos: 8       # guaranteed, non-preemptible
  pro6000_qos: 4
```

Partitions, GPU types, and allowed QoS lists are auto-discovered from the cluster via `scontrol`. VRAM is a built-in lookup table. You only need to declare QoS limits that affect your strategy.

The `shell` setting controls which shell runs your commands on GPU nodes. Set it to match whatever shell has your conda/micromamba/module setup.

State and job data are stored in `~/.local/share/autolease/`.

## CLI reference

### Pool management

| Command | Description |
|---|---|
| `autolease up <partition> [-n GPUs] [-t TIME] [-q QOS]` | Acquire a lease. QoS auto-selected if omitted. Time defaults to partition max. |
| `autolease down` | Release all leases |
| `autolease pool` | Show leases with remaining time, detect lost leases |
| `autolease check [--replace]` | Quick health-check. `--replace` auto-swaps bad ones. |
| `autolease test` | Thorough GPU test (nvidia-smi details + CUDA compute cap) |
| `autolease renew [-t MINUTES]` | Renew leases within N minutes of expiry (default: 30) |
| `autolease bad-nodes [--clear]` | Show or reset the bad-node list |
| `autolease shell [lease_id] [-g TYPE] [-n GPUs] [-s SHELL]` | Open an interactive shell on a lease via `srun --pty` |

### Job queue

All job commands accept an optional job ID. If omitted, they read `AUTOLEASE_JOB_ID` from the environment, or fall back to the last submitted job.

| Command | Description |
|---|---|
| `autolease run [opts] -- <command>` | Submit a job (prints job ID to stdout) |
| `autolease poll [id] [-i SECS]` | Tail stdout, refreshing periodically (default: 10s) |
| `autolease status [id] [--json]` | One-word state: `queued`, `running`, `done:0`, `failed` |
| `autolease jobs [project] [-a]` | List jobs, optionally filtered by project |
| `autolease log [id] [--stderr] [-n LINES]` | Read job output |
| `autolease wait <id>` | Block until done, print output, exit with job's code |
| `autolease cancel [id]` | Kill a queued or running job |
| `autolease redo [id] [--poll]` | Re-submit the same command as a previous job |

**`run` options:**

- `-p, --project` Project name (default: auto-detected from git root or cwd)
- `-g, --gpu-type` Require exact GPU type (e.g. `A6000`)
- `-n, --num-gpus` Number of GPUs (default: 1)
- `--min-vram` Minimum VRAM per GPU in GB (e.g. `48`)
- `-P, --priority` Job priority (default: 0). Higher priority jobs preempt lower ones.
- `-e, --env` Conda/micromamba env (overrides config default)
- `--no-sync` Skip code sync before submitting
- `--poll` Auto-start polling after submit

### Code sync

| Command | Description |
|---|---|
| `autolease sync [--dry-run]` | Manually sync code files to cluster |
| `autolease pull <path>` | Pull files from cluster (e.g. `results/`) |

Code files (`*.py`, `*.yaml`, `*.sh`, etc.) are auto-synced before each `run`. Remote path mirrors your local path relative to `~`. Only files newer locally are transferred. Data, checkpoints, `.git`, and other non-code files are excluded.

### Events

| Command | Description |
|---|---|
| `autolease events [-n N]` | Show last N dispatch/preemption events (default: 20) |

### Info

| Command | Description |
|---|---|
| `autolease nodes` | Show all cluster nodes with GPU type, count, state |
| `autolease partitions` | Show partition/QoS/VRAM reference table |

### TUI

| Command | Description |
|---|---|
| `autolease` | Launch interactive dashboard |
| `autolease tui` | Same as above |

**TUI keybindings:**

| Key | Action |
|---|---|
| `h/j/k/l` | Navigate: focus left panel / cursor down / cursor up / focus right panel |
| `a` | Acquire lease (opens modal with per-node availability) |
| `d` | Release selected lease (with confirmation) |
| `D` | Release all leases (with confirmation) |
| `e` | Toggle stdout/stderr of selected job |
| `c` | Cancel selected lease or job depending on focus (with confirmation) |
| `H` | Health-check all leases |
| `r` | Force refresh |
| `q` | Quit |

Selecting a job (arrow keys, j/k, or mouse) auto-loads its log in the output panel. Panel titles show aggregate stats (QoS usage, running job count, GPUs used).

## QoS strategy

QoS is auto-selected based on current usage and configured limits:

| QoS | Default limit | Behavior |
|---|---|---|
| `base_qos` | 8 GPUs/user | Guaranteed, non-preemptible. Used first. |
| `big_qos` | Unlimited | Preemptible. Overflow when base_qos is full. |
| `a100_qos` | Unlimited | Required for A100 partitions. |
| `pro6000_qos` | 4 GPUs/user | Required for PRO6000 partitions. |

Limits are configured in `config.yaml` under `qos:`. You can override with `-q <qos>` on `autolease up`.

## Architecture

```
autolease/
  cli.py       CLI entry point (argparse)
  tui.py       Interactive dashboard (textual)
  config.py    Configuration, QoS strategy, partition discovery
  pool.py      Lease lifecycle: acquire, release, health-check, renew
  queue.py     Async job queue, dispatcher, remote execution
  slurm.py     Low-level Slurm commands over SSH
  sync.py      Code file sync via rsync
```

- **Leases** are `sbatch --wrap 'sleep infinity'` jobs. Work runs inside them via `srun --jobid --overlap`.
- **Jobs** execute on the remote via `nohup srun ... &`, surviving SSH disconnects. Output is written to `~/.autolease/jobs/<id>/` on the cluster.
- **State** is local JSON files. No database, no daemon.
- **Dispatch** is opportunistic: write-path commands (`run`, `cancel`, `up`, `down`) and the TUI refresh dispatch pending jobs. Read-only commands (`status`, `log`, `jobs`) never dispatch — they only do the SSH calls they actually need. Jobs prefer the smallest VRAM lease that fits.
- **Priority preemption**: when a high-priority job has no free lease, it preempts the lowest-priority running job on a matching lease. The victim is re-queued. All events are logged.
- **Code sync**: rsync with an allowlist of code file extensions. Only newer files are transferred. Remote path mirrors local `~/` structure.
- **Env activation**: commands are wrapped with `env_activate` template (e.g. `micromamba run -n dl <command>`).
- **Lease discovery**: `refresh()` scans `squeue` in a single SSH call and adopts any autolease-named jobs not in local state.
- **Partition discovery**: partitions, GPU types, and QoS lists are auto-discovered from `scontrol show partition` + `sinfo`.
- **SSH connection reuse**: every `ssh` and `rsync` autolease runs goes through an OpenSSH ControlMaster connection (socket at `$XDG_RUNTIME_DIR/autolease-cm-%C`, 10-minute persistence). The first call establishes a TCP/auth handshake; subsequent calls multiplex over the same connection (~5ms each). The TUI refresh, `autolease poll`, and concurrent commands all share one connection per cluster.
- **Per-job SSH cost**: `_check_remote` checks PID liveness and exit code in one combined shell command (1 SSH per running job, not 2–3). The dispatcher short-circuits before any SSH calls when the queue is empty.

## License

MIT
