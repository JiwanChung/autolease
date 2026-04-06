# autolease

Personal GPU pool manager for shared Slurm clusters.

autolease holds GPU allocations as persistent Slurm jobs and lets you (or your coding agents) submit work to them without fighting for resources each time. It handles QoS selection, bad-node detection, lease renewal, and job queuing across multiple projects.

## Why

On a competitive shared GPU cluster:

- **Allocations are slow to get.** You don't want to `salloc` for every experiment.
- **Nodes are unreliable.** CUDA versions, driver states, and hardware health vary across nodes.
- **QoS rules are complex.** Guaranteed slots are limited; preemptible slots are unlimited but risky.
- **Multiple projects compete.** You run 8-10 projects and they shouldn't block each other.

autolease solves this by maintaining a personal pool of held GPU allocations and routing work to them.

## How it works

1. **Leases** are persistent `sbatch sleep` jobs that hold GPUs. You acquire them with `autolease up <partition>`.
2. **Jobs** are submitted to leases via `srun --jobid --overlap`. They're async by default and queued across projects with round-robin fairness.
3. **Health checks** run `nvidia-smi` on each lease. Bad nodes are auto-excluded.
4. **QoS auto-selection** fills guaranteed slots first (`base_qos`), then overflows to preemptible (`big_qos`).
5. **Preemption detection** notifies you when a lease disappears unexpectedly.

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

# Submit work
id=$(autolease run -- python train.py)
autolease status $id                   # queued / running / done:0 / failed
autolease log $id                      # read stdout
autolease wait $id                     # block until done, print output

# Done for the day
autolease down                         # release all leases
```

## Configuration

Config lives at `~/.config/autolease/config.yaml`:

```yaml
ssh_host: your-cluster

exclude_nodes:
  - bad-node-01

# QoS GPU limits per user (0 or omitted = unlimited)
qos:
  base_qos: 8       # guaranteed, non-preemptible
  pro6000_qos: 4
```

That's it. Partitions, GPU types, and allowed QoS lists are auto-discovered from the cluster via `scontrol`. VRAM is a built-in lookup table. You only need to declare QoS limits that affect your strategy.

State and job data are stored in `~/.local/share/autolease/`.

## CLI reference

### Pool management

| Command | Description |
|---|---|
| `autolease up <partition> [-n GPUs] [-t TIME] [-q QOS]` | Acquire a lease. QoS auto-selected if omitted. Time defaults to partition max. |
| `autolease down` | Release all leases |
| `autolease pool` | Show leases with remaining time, detect lost leases |
| `autolease check [--replace]` | Health-check leases. `--replace` auto-swaps bad ones. |
| `autolease renew [-t MINUTES]` | Renew leases within N minutes of expiry (default: 30) |
| `autolease bad-nodes [--clear]` | Show or reset the bad-node list |

### Job queue

| Command | Description |
|---|---|
| `autolease run [opts] -- <command>` | Submit a job (prints job ID) |
| `autolease status <id> [--json]` | One-word state: `queued`, `running`, `done:0`, `failed` |
| `autolease jobs [project] [-a]` | List jobs, optionally filtered by project |
| `autolease log <id> [--stderr] [-n LINES]` | Read job output |
| `autolease wait <id>` | Block until done, print output, exit with job's code |
| `autolease cancel <id>` | Kill a queued or running job |

**`run` options:**

- `-p, --project` Project name (default: auto-detected from git root or cwd)
- `-g, --gpu-type` Require exact GPU type (e.g. `A6000`)
- `-n, --num-gpus` Number of GPUs (default: 1)
- `--min-vram` Minimum VRAM per GPU in GB (e.g. `48`)

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
| `a` | Acquire lease (opens modal with per-node availability) |
| `d` | Release selected lease |
| `D` | Release all leases |
| `l` | View stdout of selected job |
| `e` | View stderr of selected job |
| `c` | Cancel selected job |
| `h` | Health-check all leases |
| `r` | Force refresh |
| `q` | Quit |

## QoS strategy

QoS is auto-selected based on current usage:

| QoS | Limit | Behavior |
|---|---|---|
| `base_qos` | 8 GPUs/user | Guaranteed, non-preemptible. Used first. |
| `big_qos` | Unlimited | Preemptible. Overflow when base_qos is full. |
| `a100_qos` | Unlimited | Required for A100 partitions. |
| `pro6000_qos` | 4 GPUs/user | Required for PRO6000 partitions. |

You can override with `-q <qos>` on `autolease up`.

## Architecture

```
autolease/
  cli.py       CLI entry point (argparse)
  tui.py       Interactive dashboard (textual)
  config.py    Configuration, QoS strategy, partition map
  pool.py      Lease lifecycle: acquire, release, health-check, renew
  queue.py     Async job queue, dispatcher, remote execution
  slurm.py     Low-level Slurm commands over SSH
```

- **Leases** are `sbatch --wrap 'sleep infinity'` jobs. Work runs inside them via `srun --jobid --overlap`.
- **Jobs** execute on the remote via `nohup srun ... &`, surviving SSH disconnects. Output is written to `~/.autolease/jobs/<id>/` on the cluster.
- **State** is local JSON files. No database, no daemon.
- **Dispatch** is opportunistic: every CLI call that touches the queue also dispatches pending jobs to free lease slots.

## License

MIT
