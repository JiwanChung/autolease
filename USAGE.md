# autolease usage for coding agents

This document describes how a coding agent (e.g. Claude Code) should use autolease to run GPU workloads on a Slurm cluster.

## Setup

autolease must be installed on the local machine (not the cluster). It communicates with the cluster via SSH.

```bash
uv tool install git+https://github.com/JiwanChung/autolease
```

Config is at `~/.config/autolease/config.yaml`. Required fields: `ssh_host`. Recommended: `shell`, `env`, `env_activate`.

## Before running jobs: ensure leases exist

Jobs need running leases (held GPU allocations) to dispatch to. Check with:

```bash
autolease pool
```

If no leases are active, acquire one:

```bash
autolease up <partition>
# Example: autolease up suma_rtx4090
# Example: autolease up suma_a6000 -n 2
```

Use `autolease partitions` to see available partitions and GPU types.
Use `autolease nodes` to see per-node GPU availability.

QoS is auto-selected. Time defaults to partition max. No other config needed.

## Submitting jobs

Every `run` is async. It returns a bare job ID on stdout. Code files are auto-synced to the cluster before dispatch. The configured conda/micromamba env is auto-activated.

```bash
# Capture the job ID for this shell
export AUTOLEASE_JOB_ID=$(autolease run -- python train.py --lr 0.001)

# Or submit and immediately tail output
autolease run --poll -- python train.py --lr 0.001
```

The command runs in the synced project directory on the cluster, with the configured env active. You don't need to `cd`, `rsync`, or `conda activate` manually.

Once `AUTOLEASE_JOB_ID` is exported, subsequent shortcut commands work without a job ID:

```bash
autolease poll         # tail stdout, refreshes every 10s
autolease log          # read stdout
autolease log --stderr # read stderr
autolease status       # one-word state
autolease cancel       # kill it
autolease redo         # re-submit the same command
```

The env var is per-shell. A different terminal has its own `AUTOLEASE_JOB_ID`. If the var isn't set, these commands fail with a helpful message — they never fall back to some "recent" global job.

### Resource requirements

```bash
# Require 2 GPUs
id=$(autolease run -n 2 -- torchrun --nproc_per_node=2 train.py)

# Require at least 48GB VRAM per GPU (matches A6000, RTX6000ADA, A100)
id=$(autolease run --min-vram 48 -- python big_model.py)

# Require exact GPU type
id=$(autolease run -g a100 -- python train.py)
```

If no matching lease is available, the job stays queued until one becomes free. Jobs without specific GPU requirements are dispatched to the smallest available GPU first, keeping large GPUs free for jobs that need them.

### Priority

Jobs have a priority (default 0). Higher-priority jobs preempt lower-priority running jobs if no free lease is available. Preempted jobs are re-queued, not lost.

```bash
# Normal training (default priority 0)
id=$(autolease run -- python train.py)

# Urgent debug run (priority 10) — will preempt a priority-0 job if needed
id=$(autolease run -P 10 -- python debug.py)
```

Use higher priority for interactive debugging. Use default (0) for batch training.

### Env override

If you need a different env than the config default:

```bash
id=$(autolease run --env myenv -- python train.py)
```

### Skip sync

If the code is already on the cluster or you're running a non-project command:

```bash
id=$(autolease run --no-sync -- nvidia-smi)
```

### Project isolation

Project is auto-detected from the git root of the current directory. Jobs from different projects are scheduled round-robin and never interfere. To override:

```bash
id=$(autolease run -p my-project -- python test.py)
```

## Checking job state

```bash
autolease status           # uses AUTOLEASE_JOB_ID
autolease status $id       # or pass explicitly
```

Returns one word:
- `queued` — waiting for a GPU slot
- `running` — executing on a GPU
- `done:0` — finished successfully (exit code 0)
- `done:1` — finished with error (exit code 1)
- `failed` — could not launch or was cancelled

For full details:

```bash
autolease status --json
```

## Reading output

```bash
autolease log              # stdout of $AUTOLEASE_JOB_ID
autolease log --stderr     # stderr
autolease log -n 20        # last 20 lines
autolease log $id          # or pass a job ID explicitly
```

## Tailing output live

```bash
autolease poll             # clears screen, reprints last 30 lines every 10s
autolease poll -i 5        # 5s refresh
autolease poll $id         # explicit job ID
```

`poll` uses one SSH call per cycle (same as `watch -n 10 autolease log $id -n 30`). Exits when the job finishes.

## Waiting for completion

Blocks until done, prints output, exits with the job's exit code:

```bash
autolease wait $id
```

## Cancelling jobs

```bash
autolease cancel           # cancels $AUTOLEASE_JOB_ID
autolease cancel $id
```

## Re-running a job

```bash
autolease redo             # re-submit the same command as $AUTOLEASE_JOB_ID
autolease redo $id         # or a specific old job
autolease redo --poll      # and start polling immediately
```

`redo` prints a new job ID. Capture it in the env var the same way:

```bash
export AUTOLEASE_JOB_ID=$(autolease redo)
```

## Interactive shell on a lease

```bash
autolease shell                # shell on first running lease
autolease shell 12345          # specific lease by job ID
autolease shell -g A6000       # lease with A6000 GPU
autolease shell -n 4           # request 4 GPUs from the lease
autolease shell -s zsh         # override shell (default: config.shell)
```

Opens via `ssh -t ... srun --jobid=<lease> --overlap --pty <shell>` so you land on the compute node with the GPUs available to your session.

## Code sync

Code files are auto-synced before each `run`. Only code files (`*.py`, `*.yaml`, `*.sh`, etc.) are synced — data, checkpoints, `.git` are excluded. Only files newer locally are transferred.

Remote path mirrors your local path relative to `~`. For example, local `~/projects/my-model/` syncs to `~/projects/my-model/` on the cluster.

To sync manually or preview:

```bash
autolease sync              # push code files now
autolease sync --dry-run    # show what would be synced
```

To pull results back:

```bash
autolease pull results/     # pull results/ dir from remote project
```

## Typical agent workflow

### Fast debug loop

```bash
# Write code, test on GPU, read output, iterate
# Use -P 10 so debug runs preempt background training if needed
export AUTOLEASE_JOB_ID=$(autolease run -P 10 -- python test_forward.py)
autolease wait $AUTOLEASE_JOB_ID
# ... read output, fix code ...
export AUTOLEASE_JOB_ID=$(autolease redo)
autolease wait $AUTOLEASE_JOB_ID
```

### Fire and check later

```bash
# Kick off training (default priority — can be preempted by debug runs)
export AUTOLEASE_JOB_ID=$(autolease run -- python train.py --epochs 50)
# ... do other work ...
autolease status            # check progress
autolease log -n 10         # peek at recent output
autolease poll              # or tail live
```

### Multiple projects

Each agent working in a different project directory gets automatic project isolation. If two agents submit jobs at the same time and there's only one lease, they queue fairly (round-robin by project).

## Scripting patterns

### Submit and poll

```bash
id=$(autolease run -- python experiment.py)
while true; do
    state=$(autolease status $id)
    case $state in
        done:0) echo "Success"; autolease log $id; break ;;
        done:*|failed) echo "Failed: $state"; autolease log $id --stderr; break ;;
        *) sleep 5 ;;
    esac
done
```

Or simpler, using the built-in poller:

```bash
autolease run --poll -- python experiment.py
```

### Batch submit

```bash
for lr in 0.1 0.01 0.001; do
    id=$(autolease run -p sweep -- python train.py --lr $lr)
    echo "lr=$lr -> job $id"
done
autolease jobs sweep -a   # watch progress
```

## Events

View dispatch and preemption history:

```bash
autolease events         # last 20 events
autolease events -n 50   # last 50 events
```

Example output:
```
[2026-04-06T15:30:01] SUBMIT job 8 project=diffusion-lm priority=10 gpus=1 cwd=~/projects/diffusion-lm
[2026-04-06T15:30:01] PREEMPT job 5 (priority=0, project=sweep) on node15 — re-queued
[2026-04-06T15:30:02] DISPATCH job 8 (priority=10) -> node15 (RTX3090) [preempted job 5]
```

## Error handling

- If `autolease run` returns a job ID but `status` shows `queued` for a long time, there may be no matching lease. Check `autolease pool` and acquire one if needed.
- If `status` shows `failed`, check `autolease log $id --stderr` for details.
- If a lease was preempted, `autolease pool` will report it. Acquire a new one.
- If a job was preempted by a higher-priority job, it goes back to `queued` and will re-dispatch when a slot opens.

## Reference: exit codes

| Command | Exit code |
|---|---|
| `autolease run` | 0 (job submitted) |
| `autolease status` | 0 if job exists, 1 if unknown |
| `autolease wait` | Job's exit code |
| `autolease cancel` | 0 if cancelled, 1 if not found |

## SSH load

autolease is designed to be a polite SSH client:

- All `ssh` and `rsync` calls share an OpenSSH ControlMaster connection. After the first command, everything else multiplexes over the same TCP connection (~5ms per call instead of ~500ms). The cluster's sshd sees one connection, not a flood. The control socket lives at `$XDG_RUNTIME_DIR/autolease-cm-%C` (or `~/.ssh/autolease-cm-%C`) and persists for 10 minutes after the last call.
- Read-only commands (`status`, `log`, `jobs`) never call `dispatch()` — they only make the SSH calls they need. A bare `autolease status $id` is one SSH call (`kill -0 $pid || cat exit_code`).
- The dispatcher short-circuits with zero SSH if the queue is empty.
- The TUI's 30-second refresh does one `squeue` + one combined-PID-check per running job + (optional) one `squeue` if there's anything to dispatch. No periodic log polling.
- `autolease poll` does one `tail -n 30` per cycle (default 10s) and a local-only state check.

To manually tear down the multiplexed connection:

```bash
ssh -O exit -o ControlPath=$XDG_RUNTIME_DIR/autolease-cm-%C <host>
```
