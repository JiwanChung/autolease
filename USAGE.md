# autolease usage for coding agents

This document describes how a coding agent (e.g. Claude Code) should use autolease to run GPU workloads on the Slurm cluster.

## Setup

autolease must be installed on the local machine (not the cluster). It communicates with the cluster via SSH.

```bash
uv tool install git+https://github.com/JiwanChung/autolease
```

Config is at `~/.config/autolease/config.yaml`. The only required field is `ssh_host`.

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

Every `run` is async. It returns a bare job ID on stdout.

```bash
id=$(autolease run -- python train.py --lr 0.001)
```

### Resource requirements

```bash
# Require 2 GPUs
id=$(autolease run -n 2 -- torchrun --nproc_per_node=2 train.py)

# Require at least 48GB VRAM per GPU (matches A6000, RTX6000ADA, A100)
id=$(autolease run --min-vram 48 -- python big_model.py)

# Require exact GPU type
id=$(autolease run -g a100 -- python train.py)
```

If no matching lease is available, the job stays queued until one becomes free.

### Project isolation

Project is auto-detected from the git root of the current directory. Jobs from different projects are scheduled round-robin and never interfere. To override:

```bash
id=$(autolease run -p my-project -- python test.py)
```

## Checking job state

```bash
autolease status $id
```

Returns one word:
- `queued` — waiting for a GPU slot
- `running` — executing on a GPU
- `done:0` — finished successfully (exit code 0)
- `done:1` — finished with error (exit code 1)
- `failed` — could not launch or was cancelled

For full details:

```bash
autolease status $id --json
```

## Reading output

```bash
autolease log $id              # stdout
autolease log $id --stderr     # stderr
autolease log $id -n 20        # last 20 lines
```

## Waiting for completion

Blocks until done, prints output, exits with the job's exit code:

```bash
autolease wait $id
```

## Cancelling jobs

```bash
autolease cancel $id
```

## Typical agent workflow

### Fast debug loop

```bash
# Write code, test on GPU, read output, iterate
id=$(autolease run -- python test_forward.py)
autolease wait $id
# ... read output, fix code ...
id=$(autolease run -- python test_forward.py)
autolease wait $id
```

### Fire and check later

```bash
# Kick off training
id=$(autolease run -- python train.py --epochs 50)
# ... do other work ...
autolease status $id        # check progress
autolease log $id -n 10     # peek at recent output
autolease wait $id          # block when ready to see results
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

### Batch submit

```bash
for lr in 0.1 0.01 0.001; do
    id=$(autolease run -p sweep -- python train.py --lr $lr)
    echo "lr=$lr -> job $id"
done
autolease jobs sweep -a   # watch progress
```

## Error handling

- If `autolease run` returns a job ID but `status` shows `queued` for a long time, there may be no matching lease. Check `autolease pool` and acquire one if needed.
- If `status` shows `failed`, check `autolease log $id --stderr` for details.
- If a lease was preempted, `autolease pool` will report it. Acquire a new one.

## Reference: exit codes

| Command | Exit code |
|---|---|
| `autolease run` | 0 (job submitted) |
| `autolease status` | 0 if job exists, 1 if unknown |
| `autolease wait` | Job's exit code |
| `autolease cancel` | 0 if cancelled, 1 if not found |
