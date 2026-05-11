"""CLI entry point for autolease."""

import argparse
import json
import os
import shlex
import sys
import time

from .config import (
    load_config, LeaseSpec, PARTITION_INFO, GPU_VRAM, QOS_GPU_LIMITS,
    pick_qos, discover_partitions, apply_qos_config,
)
from .pool import Pool
from .queue import JobQueue


# ── Pool commands ──

def cmd_up(args):
    cfg = load_config(args.config)
    pool = Pool(cfg)

    # Auto-select QoS if not specified
    qos = args.qos
    if qos is None:
        usage = pool.slurm.gpu_usage_by_qos()
        qos = pick_qos(args.partition, args.num_gpus, usage)
        limit = QOS_GPU_LIMITS.get(qos, 0)
        current = usage.get(qos, 0)
        limit_str = f"{current}/{limit}" if limit else f"{current}/inf"
        print(f"QoS: {qos} (usage: {limit_str} GPUs)")

    spec = LeaseSpec(
        partition=args.partition,
        qos=qos,
        num_gpus=args.num_gpus,
        time=args.time,
    )
    time_str = spec.time or "max"
    print(f"Acquiring {spec.partition}/{spec.gpu_type} x{spec.num_gpus} ({time_str})...")
    try:
        lease = pool.up(spec)
    except RuntimeError as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  job {lease.job_id} submitted. Waiting...")
    ok = pool.wait_and_check(lease, poll_interval=3, max_wait=120)
    if ok:
        print(f"  job {lease.job_id} on {lease.node}: OK")
    else:
        print(f"  job {lease.job_id} on {lease.node or '?'}: FAIL")
        if lease.node:
            print(f"  -> marking {lease.node} as bad, replacing...")
            pool.check_and_replace()


def cmd_down(args):
    pool = Pool(load_config(args.config))
    n = pool.down()
    print(f"Cancelled {n} lease(s).")


def cmd_pool_status(args):
    pool = Pool(load_config(args.config))
    leases = pool.status()

    if pool.lost_leases:
        for l in pool.lost_leases:
            print(f"LOST: lease {l.job_id} on {l.node or '?'} ({l.partition}/{l.gpu_type} x{l.num_gpus}) — preempted or expired")
        print()

    if not leases:
        print("No active leases.")
        return

    print(f"{'JOB_ID':>10}  {'PARTITION':<22}  {'GPU':<14}  {'#':>2}  {'NODE':<12}  {'STATE':<10}  {'REMAINING':<10}")
    print("-" * 90)
    for l in leases:
        remaining = pool.remaining_minutes(l)
        if remaining is not None:
            rem_str = f"{remaining/60:.1f}h" if remaining >= 60 else f"{remaining:.0f}m"
        else:
            rem_str = "—"
        print(f"{l.job_id:>10}  {l.partition:<22}  {l.gpu_type:<14}  {l.num_gpus:>2}  {l.node or '—':<12}  {l.state:<10}  {rem_str:<10}")

    bad = pool.bad_nodes()
    if bad:
        print(f"\nBad nodes: {', '.join(bad)}")


def cmd_check(args):
    pool = Pool(load_config(args.config))
    if args.replace:
        actions = pool.check_and_replace()
        if not actions:
            print("No active leases to check.")
            return
        for a in actions:
            node = a.get("node", "?")
            if a["action"] == "ok":
                print(f"  job {a['job_id']} on {node}: OK")
            elif a["action"] == "bad":
                print(f"  job {a['job_id']} on {node}: FAIL — {a['reason']}")
            elif a["action"] == "replacement":
                print(f"  job {a['job_id']}: replacement submitted ({a['reason']})")
            elif a["action"] == "skip":
                print(f"  job {a['job_id']}: {a['reason']} (skipped)")
    else:
        leases = pool.status()
        if not leases:
            print("No active leases to check.")
            return
        for l in leases:
            if l.state != "RUNNING":
                print(f"  job {l.job_id} ({l.partition}): {l.state} (skipped)")
                continue
            ok = pool.check_lease(l)
            mark = "OK" if ok else "FAIL"
            print(f"  job {l.job_id} on {l.node}: {mark}")

    bad = pool.bad_nodes()
    if bad:
        print(f"\nBad nodes: {', '.join(bad)}")


def cmd_test(args):
    pool = Pool(load_config(args.config))
    leases = pool.status()
    if not leases:
        print("No active leases to test.")
        return
    for l in leases:
        if l.state != "RUNNING":
            print(f"lease {l.job_id} ({l.partition}): {l.state} — skipped")
            continue
        print(f"lease {l.job_id} on {l.node} ({l.gpu_type} x{l.num_gpus}):")
        result = pool.test_lease(l)
        # nvidia-smi
        smi = result.get("nvidia_smi", {})
        gpus = smi.get("gpus", [])
        if gpus:
            for g in gpus:
                print(f"  smi: {g['name']}  {g['mem_total_mb']}MB total  "
                      f"{g['mem_free_mb']}MB free  {g['temp_c']}C  "
                      f"driver={g['driver']}")
        else:
            print("  smi: FAILED")
        # cuda
        cuda = result.get("cuda", {})
        if cuda.get("ok"):
            caps = ", ".join(cuda.get("compute_caps", []))
            print(f"  cuda: {len(cuda.get('gpu_list', []))} GPU(s) visible, compute cap: {caps}")
        elif cuda.get("error"):
            print(f"  cuda: {cuda['error'][:80]}")
        # errors
        for e in result.get("errors", []):
            print(f"  ERROR: {e}")
        print(f"  -> {'PASS' if result['ok'] else 'FAIL'}")
        print()


def cmd_renew(args):
    pool = Pool(load_config(args.config))
    print(f"Checking leases (threshold: {args.threshold}min)...")
    actions = pool.renew(threshold_minutes=args.threshold)
    if not actions:
        print("No active leases.")
        return
    for a in actions:
        if a["action"] == "ok":
            print(f"  job {a['job_id']}: {a['remaining_min']}min remaining — OK")
        elif a["action"] == "renewing":
            print(f"  job {a['job_id']}: {a['remaining_min']}min remaining — renewing...")
        elif a["action"] == "renewed":
            print(f"  job {a['job_id']} on {a.get('node', '?')}: renewed ({a['reason']})")
        elif a["action"] == "renewal_pending":
            print(f"  job {a['job_id']}: {a['reason']}")
        elif a["action"] == "renew_failed":
            print(f"  job {a['job_id']}: FAILED — {a['reason']}")
        elif a["action"] == "skip":
            print(f"  job {a['job_id']}: skipped — {a['reason']}")


def cmd_bad_nodes(args):
    pool = Pool(load_config(args.config))
    if args.clear:
        pool.clear_bad_nodes()
        print("Cleared dynamic bad-node list.")
        return
    bad = pool.bad_nodes()
    if bad:
        print("Bad nodes:")
        for n in bad:
            src = "config" if n in pool.config.exclude_nodes else "detected"
            print(f"  {n} ({src})")
    else:
        print("No bad nodes.")


# ── Job commands ──

def _get_job_id(args, cfg) -> int:
    """Get a single job ID from args or AUTOLEASE_JOB_ID env var.
    Kept for backward compat / commands that genuinely take one job."""
    jid = getattr(args, "job_id", None)
    if jid is not None:
        return jid
    env_id = os.environ.get("AUTOLEASE_JOB_ID")
    if env_id:
        return int(env_id)
    print(
        "No job ID. Pass one, or set AUTOLEASE_JOB_ID in your shell:\n"
        "  export AUTOLEASE_JOB_ID=$(autolease run -- <command>)",
        file=sys.stderr,
    )
    sys.exit(1)


def _get_job_ids(args, cfg) -> list[int]:
    """Get one or more job IDs from `job_ids` (nargs='*') or env var."""
    ids = getattr(args, "job_ids", None) or []
    if ids:
        return ids
    env_id = os.environ.get("AUTOLEASE_JOB_ID")
    if env_id:
        return [int(env_id)]
    print(
        "No job ID(s). Pass one or more, or set AUTOLEASE_JOB_ID in your shell:\n"
        "  export AUTOLEASE_JOB_ID=$(autolease run -- <command>)",
        file=sys.stderr,
    )
    sys.exit(1)


def cmd_run(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    cmd_parts = [c for c in args.command if c != "--"]
    command = " ".join(cmd_parts)
    if not command:
        print("Error: no command specified", file=sys.stderr)
        sys.exit(1)
    job = q.submit(
        command=command,
        project=args.project,
        num_gpus=args.num_gpus,
        min_vram=args.min_vram,
        gpu_type=args.gpu_type,
        priority=args.priority,
        env=args.env,
        no_sync=args.no_sync,
    )
    print(job.id)
    if args.poll:
        _do_poll(q, job.id)


def cmd_status(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    job_ids = _get_job_ids(args, cfg)
    show_id = len(job_ids) > 1
    for jid in job_ids:
        # Read-only: refresh remote state (1 SSH) but don't dispatch
        job = q.get(jid, dispatch=False)
        if job is None:
            line = "unknown"
        elif args.json:
            from dataclasses import asdict
            line = json.dumps(asdict(job))
        elif job.state == "done":
            line = f"done:{job.exit_code}"
        else:
            line = job.state
        if show_id:
            print(f"{jid}\t{line}")
        else:
            print(line)


def cmd_jobs(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    # Read-only: refresh remote state but don't dispatch
    jobs = q.list_jobs(project=args.project, active_only=args.active,
                       dispatch=False)
    if not jobs:
        print("No jobs.")
        return
    print(f"{'ID':>5}  {'PRI':>3}  {'STATE':<10}  {'PROJECT':<20}  {'GPU':>4}  {'VRAM':>4}  {'NODE':<10}  {'CMD':<28}")
    print("-" * 97)
    for j in jobs:
        state = f"done:{j.exit_code}" if j.state == "done" else j.state
        cmd_short = j.command[:26] + ".." if len(j.command) > 28 else j.command
        vram = f"{j.min_vram}G" if j.min_vram else "—"
        print(f"{j.id:>5}  {j.priority:>3}  {state:<10}  {j.project:<20}  {j.num_gpus:>4}  {vram:>4}  {j.node or '—':<10}  {cmd_short:<28}")


def cmd_log(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    job_id = _get_job_id(args, cfg)
    stream = "stderr" if args.stderr else "stdout"
    # 1 SSH call — just tail the log
    out = q.read_log(job_id, stream=stream, tail=args.tail)
    if out:
        print(out, end="")
    else:
        # Fallback: local-only load to report state
        job = q.get(job_id, refresh=False, dispatch=False)
        if job is None:
            print(f"Job {job_id} not found.", file=sys.stderr)
        elif job.state == "queued":
            print(f"Job {job_id} is queued (no output yet).", file=sys.stderr)


def cmd_wait(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    while True:
        job = q.get(args.job_id)
        if job is None:
            print(f"Job {args.job_id} not found.", file=sys.stderr)
            sys.exit(1)
        if job.state in ("done", "failed"):
            out = q.read_log(args.job_id, stream="stdout")
            err = q.read_log(args.job_id, stream="stderr")
            if out:
                print(out, end="")
            if err:
                print(err, end="", file=sys.stderr)
            sys.exit(job.exit_code or 0)
        time.sleep(args.poll)


def _do_poll(q: JobQueue, job_id: int, interval: float = 10.0, tail_n: int = 30):
    """Polling loop: tail combined stdout+stderr + check for exit_code file
    in a SINGLE SSH call per cycle. Exits when exit_code appears (job done)."""
    rdir = f"~/.autolease/jobs/{job_id}"
    # Combined command:
    #   tail of combined file + separator + exit_code file (if present)
    # Falls back to stdout file if combined doesn't exist (old jobs).
    inner = (
        f"if [ -e {rdir}/combined ]; then "
        f"  tail -n {tail_n} {rdir}/combined 2>/dev/null; "
        f"else "
        f"  tail -n {tail_n} {rdir}/stdout 2>/dev/null; "
        f"fi; "
        f"echo __AL_POLL_SEP__; "
        f"cat {rdir}/exit_code 2>/dev/null"
    )
    cmd = f"/bin/sh -c {shlex.quote(inner)}"
    try:
        while True:
            r = q.slurm.cfg.run(cmd, timeout=15)
            text = r.stdout
            sep_idx = text.rfind("__AL_POLL_SEP__")
            if sep_idx >= 0:
                log_part = text[:sep_idx].rstrip("\n")
                exit_part = text[sep_idx + len("__AL_POLL_SEP__"):].strip()
            else:
                log_part = text
                exit_part = ""

            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            if log_part:
                print(log_part, flush=True)

            if exit_part:
                # exit_code file exists → job done. Update local state
                # so future commands see the right thing.
                try:
                    exit_code = int(exit_part)
                except ValueError:
                    exit_code = None
                job = q._load_job(job_id)
                if job and job.state == "running":
                    job.state = "done"
                    job.exit_code = exit_code
                    from autolease.queue import _now as _t
                    job.finished = _t()
                    q._save_job(job)
                    q._log_job_history(job_id, "DONE", exit_code=exit_code)
                print(f"\n--- job {job_id} done (exit {exit_code}) ---",
                      file=sys.stderr)
                sys.exit(exit_code or 0)

            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n--- stopped polling job {job_id} ---", file=sys.stderr)


def cmd_poll(args):
    """Tail stdout and stderr of a job, refreshing periodically."""
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    job_id = _get_job_id(args, cfg)
    interval = args.interval if args.interval is not None else cfg.poll_interval
    tail_n = args.tail if args.tail is not None else cfg.poll_tail_lines
    _do_poll(q, job_id, interval=interval, tail_n=tail_n)


def cmd_cancel(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    job_ids = _get_job_ids(args, cfg)
    failures = 0
    for jid in job_ids:
        if q.cancel(jid):
            print(f"Cancelled job {jid}.")
        else:
            print(f"Job {jid} not found or already finished.", file=sys.stderr)
            failures += 1
    if failures:
        sys.exit(1)


def cmd_redo(args):
    """Re-submit the same command as a previous job."""
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    job_id = _get_job_id(args, cfg)
    old_job = q._load_job(job_id)
    if old_job is None:
        print(f"Job {job_id} not found.", file=sys.stderr)
        sys.exit(1)
    new_job = q.submit(
        command=old_job.command,
        project=old_job.project,
        num_gpus=old_job.num_gpus,
        min_vram=old_job.min_vram,
        gpu_type=old_job.gpu_type,
        priority=old_job.priority,
        no_sync=False,
    )
    print(new_job.id)
    if args.poll:
        _do_poll(q, new_job.id)


SHELL_INIT_BASH = """\
# autolease shell helpers — eval $(autolease shell-init bash) in your rc file
# Captures the job ID from `autolease run` into AUTOLEASE_JOB_ID for this shell.
al-run() {
    local id
    id=$(autolease run "$@") || return $?
    export AUTOLEASE_JOB_ID="$id"
    echo "AUTOLEASE_JOB_ID=$id" >&2
}
al-redo() {
    local id
    id=$(autolease redo "$@") || return $?
    export AUTOLEASE_JOB_ID="$id"
    echo "AUTOLEASE_JOB_ID=$id" >&2
}
"""

SHELL_INIT_ZSH = SHELL_INIT_BASH  # bash function syntax works in zsh

SHELL_INIT_FISH = """\
# autolease shell helpers — `autolease shell-init fish | source` in config.fish
function al-run
    set -l id (autolease run $argv)
    or return $status
    set -gx AUTOLEASE_JOB_ID $id
    echo "AUTOLEASE_JOB_ID=$id" >&2
end
function al-redo
    set -l id (autolease redo $argv)
    or return $status
    set -gx AUTOLEASE_JOB_ID $id
    echo "AUTOLEASE_JOB_ID=$id" >&2
end
"""


def cmd_shell_init(args):
    """Print shell helper functions to stdout for sourcing."""
    snippets = {
        "bash": SHELL_INIT_BASH,
        "zsh": SHELL_INIT_ZSH,
        "fish": SHELL_INIT_FISH,
    }
    snippet = snippets.get(args.shell)
    if snippet is None:
        print(f"Unknown shell: {args.shell}. Supported: {', '.join(snippets)}",
              file=sys.stderr)
        sys.exit(1)
    print(snippet, end="")


def cmd_info(args):
    """One-shot dump of everything autolease knows about a job:
    state, lease, node, files, history, recent log."""
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    job_ids = _get_job_ids(args, cfg)
    for jid in job_ids:
        j = q._load_job(jid)
        if j is None:
            print(f"Job {jid}: not found", file=sys.stderr)
            continue
        print(f"━━━ job {j.id} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  project    : {j.project}")
        print(f"  command    : {j.command}")
        print(f"  state      : {j.state}" + (f" (exit {j.exit_code})" if j.exit_code is not None else ""))
        print(f"  num_gpus   : {j.num_gpus}" + (f"  min_vram: {j.min_vram}G" if j.min_vram else ""))
        if j.gpu_type:
            print(f"  gpu_type   : {j.gpu_type}")
        print(f"  priority   : {j.priority}")
        print(f"  lease      : {j.lease_job_id}")
        print(f"  step_name  : {j.step_name}")
        print(f"  node       : {j.node}")
        print(f"  remote_pid : {j.remote_pid}")
        print(f"  remote_cwd : {j.remote_cwd}")
        print(f"  submitted  : {j.submitted}")
        print(f"  started    : {j.started}")
        print(f"  finished   : {j.finished}")
        print()
        # History
        history = q._read_job_history(jid)
        if history:
            print(f"  history ({len(history)} events):")
            for line in history:
                print(f"    {line}")
        else:
            print(f"  history: (no events recorded — pre-history job)")
        print()
        # Files on remote
        rdir = f"~/.autolease/jobs/{jid}"
        print(f"  remote dir : {rdir}")
        try:
            r = q.slurm.cfg.run(
                f"ls -la {rdir}/ 2>/dev/null | tail -n +2",
                timeout=10,
            )
            for line in r.stdout.strip().splitlines():
                print(f"    {line}")
        except Exception as e:
            print(f"    (ls failed: {e})")
        print()
        # Show last few lines of log
        if not args.no_log:
            tail_n = args.tail
            print(f"  log tail ({tail_n} lines from combined):")
            log = q.read_log(jid, stream="combined", tail=tail_n)
            if not log:
                log = q.read_log(jid, stream="stdout", tail=tail_n) or "(no output)"
            for line in log.rstrip("\n").splitlines()[-tail_n:]:
                print(f"    {line}")
        print()


def cmd_recover(args):
    """Re-check jobs marked failed/done. If the wrapper is still alive on
    the cluster (transient SSH glitch caused a wrong LOST), restore state
    to running. Local-only: never touches the remote process."""
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    targets = []
    if args.job_id is not None:
        j = q._load_job(args.job_id)
        if j is None:
            print(f"Job {args.job_id} not found.", file=sys.stderr)
            sys.exit(1)
        targets = [j]
    else:
        targets = [j for j in q._all_jobs()
                   if j.state == "failed" and j.exit_code is None
                   and j.remote_pid]
    if not targets:
        print("No jobs to recover (none are marked failed with exit_code=None).",
              file=sys.stderr)
        return
    for j in targets:
        state, code = q._check_remote(j)
        if state == "running":
            j.state = "running"
            j.finished = None
            j.exit_code = None
            q._save_job(j)
            q._log_job_history(j.id, "RECOVER", to="running")
            print(f"  job {j.id}: recovered (wrapper PID {j.remote_pid} still alive)")
        elif state == "done":
            j.state = "done"
            j.exit_code = code
            q._save_job(j)
            q._log_job_history(j.id, "RECOVER", to="done", exit_code=code)
            print(f"  job {j.id}: actually done (exit {code})")
        elif state == "lost":
            print(f"  job {j.id}: confirmed lost (PID gone, no exit_code, output quiet)")
        else:
            print(f"  job {j.id}: SSH check returned {state} — leaving alone")


def cmd_ssh_reset(args):
    """Tear down any ControlMaster connection to ssh_host.
    Use when ssh hangs after a network blip / cluster reboot."""
    from .slurm import _recover_control_socket, _control_socket_dir
    cfg = load_config(args.config)
    if not cfg.ssh_host:
        print("No ssh_host configured.", file=sys.stderr)
        sys.exit(1)
    print(f"Tearing down ControlMaster for {cfg.ssh_host}...", file=sys.stderr)
    SSH_OPTS = (
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={os.path.join(_control_socket_dir(), 'autolease-cm-%C')}",
        "-o", "ControlPersist=10m",
    )
    _recover_control_socket(SSH_OPTS, cfg.ssh_host)
    # Also force-remove ALL autolease sockets in case any are stale
    import glob
    for sock in glob.glob(os.path.join(_control_socket_dir(), "autolease-cm-*")):
        try:
            os.unlink(sock)
            print(f"  removed {sock}", file=sys.stderr)
        except OSError:
            pass
    print("Done.", file=sys.stderr)


def cmd_shell(args):
    """Open an interactive shell on a held lease via srun --pty."""
    import subprocess
    cfg = load_config(args.config)
    pool = Pool(cfg)

    if args.lease_id is not None:
        leases = [l for l in pool.status() if l.job_id == args.lease_id]
        if not leases:
            print(f"Lease {args.lease_id} not found.", file=sys.stderr)
            sys.exit(1)
        lease = leases[0]
        if lease.state != "RUNNING":
            print(f"Lease {args.lease_id} is {lease.state}, not RUNNING.", file=sys.stderr)
            sys.exit(1)
    else:
        lease = pool.find_running_lease(gpu_type=args.gpu_type, min_gpus=args.num_gpus)
        if lease is None:
            msg = "No running lease found"
            if args.gpu_type:
                msg += f" for gpu_type={args.gpu_type}"
            print(msg + ".", file=sys.stderr)
            sys.exit(1)

    shell = args.shell or cfg.shell
    print(f"Opening {shell} on lease {lease.job_id} ({lease.node} / {lease.gpu_type} x{args.num_gpus})...",
          file=sys.stderr)
    srun = (
        f"srun --jobid={lease.job_id} --gres=gpu:{args.num_gpus} --overlap"
        f" --pty {shell}"
    )
    if cfg.ssh_host:
        cmd = ["ssh", "-t", *pool.slurm.cfg.ssh_opts, cfg.ssh_host, srun]
    else:
        cmd = ["bash", "-c", srun]
    sys.exit(subprocess.call(cmd))


# ── Info commands ──

def cmd_sync(args):
    from .sync import sync as rsync_project, get_remote_dir
    cfg = load_config(args.config)
    remote = get_remote_dir(cfg)
    print(f"Syncing code files -> {cfg.ssh_host}:{remote}/")
    r = rsync_project(cfg, dry_run=args.dry_run, verbose=True, force=True)
    if r.stdout:
        print(r.stdout, end="")
    if r.returncode != 0:
        print(f"rsync failed: {r.stderr.strip()[:200]}", file=sys.stderr)
        sys.exit(1)
    if args.dry_run:
        print("(dry run — no files transferred)")
    else:
        print("Done.")


def cmd_pull(args):
    from .sync import pull as rsync_pull
    cfg = load_config(args.config)
    print(f"Pulling {args.path} from {cfg.ssh_host}...")
    r = rsync_pull(cfg, args.path, verbose=True)
    if r.stdout:
        print(r.stdout, end="")
    if r.returncode != 0:
        print(f"rsync failed: {r.stderr.strip()[:200]}", file=sys.stderr)
        sys.exit(1)
    print("Done.")


def cmd_events(args):
    cfg = load_config(args.config)
    log_file = os.path.join(cfg.state_path, "events.log")
    if not os.path.exists(log_file):
        print("No events yet.")
        return
    with open(log_file) as f:
        lines = f.readlines()
    tail = args.tail or 20
    for line in lines[-tail:]:
        print(line, end="")


def cmd_nodes(args):
    from .slurm import Slurm, SlurmConfig
    cfg = load_config(args.config)
    slurm = Slurm(SlurmConfig(ssh_host=cfg.ssh_host, shell=cfg.shell))
    nodes = slurm.sinfo_gpus()
    print(f"{'NODE':<12}  {'GPU':<14}  {'#':>2}  {'MEM_MB':>8}  {'STATE':<12}")
    print("-" * 56)
    for n in nodes:
        print(f"{n.node:<12}  {n.gpu_type:<14}  {n.gpu_count:>2}  {n.mem_mb:>8}  {n.state:<12}")


def cmd_partitions(args):
    print(f"{'PARTITION':<22}  {'GPU':<14}  {'VRAM':>6}  {'QOS':<30}")
    print("-" * 76)
    for part, (qos_list, gpu) in sorted(PARTITION_INFO.items()):
        vram = GPU_VRAM.get(gpu, "?")
        print(f"{part:<22}  {gpu:<14}  {vram:>4}GB  {','.join(qos_list):<30}")


# ── Main ──

def main():
    p = argparse.ArgumentParser(
        prog="autolease",
        description="Personal GPU pool manager for Slurm clusters",
    )
    p.add_argument("-c", "--config", default=None, help="Config file path")
    sub = p.add_subparsers(dest="cmd")

    # Pool management
    up_p = sub.add_parser("up", help="Acquire a GPU lease")
    up_p.add_argument("partition", help="Slurm partition (see `autolease partitions`)")
    up_p.add_argument("-q", "--qos", default=None,
                      help="QoS (default: auto — base_qos if room, big_qos otherwise)")
    up_p.add_argument("-n", "--num-gpus", type=int, default=1,
                      help="Number of GPUs (default: 1)")
    up_p.add_argument("-t", "--time", default=None,
                      help="Time limit (default: partition max)")

    sub.add_parser("down", help="Release all leases")
    sub.add_parser("pool", help="Show lease status with remaining time")

    check_p = sub.add_parser("check", help="Quick health-check running leases")
    check_p.add_argument("--replace", action="store_true",
                         help="Auto-replace bad leases")

    sub.add_parser("test", help="Thorough GPU test (nvidia-smi + CUDA compute)")

    renew_p = sub.add_parser("renew", help="Renew leases nearing expiry")
    renew_p.add_argument("-t", "--threshold", type=float, default=30.0,
                         help="Renew when fewer than N minutes remain (default: 30)")

    bad_p = sub.add_parser("bad-nodes", help="Show/manage bad node list")
    bad_p.add_argument("--clear", action="store_true",
                       help="Clear dynamically detected bad nodes")

    # Job queue
    run_p = sub.add_parser("run", help="Submit a job (async, prints job ID)")
    run_p.add_argument("-p", "--project", default=None,
                       help="Project name (default: auto-detect from cwd)")
    run_p.add_argument("-g", "--gpu-type", default=None,
                       help="Require exact GPU type (e.g. A6000)")
    run_p.add_argument("-n", "--num-gpus", type=int, default=1,
                       help="Number of GPUs needed (default: 1)")
    run_p.add_argument("--min-vram", type=int, default=0,
                       help="Minimum VRAM per GPU in GB (e.g. 48)")
    run_p.add_argument("-P", "--priority", type=int, default=0,
                       help="Job priority (higher preempts lower, default: 0)")
    run_p.add_argument("-e", "--env", default=None,
                       help="Conda/micromamba env (overrides config default)")
    run_p.add_argument("--no-sync", action="store_true",
                       help="Skip code sync before submitting")
    run_p.add_argument("--poll", action="store_true",
                       help="Auto-start polling after submit")
    run_p.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    status_p = sub.add_parser("status", help="Get job state (one word per job)")
    status_p.add_argument("job_ids", type=int, nargs="*",
                          help="Job IDs (default: AUTOLEASE_JOB_ID)")
    status_p.add_argument("--json", action="store_true", help="Full JSON output")

    jobs_p = sub.add_parser("jobs", help="List jobs")
    jobs_p.add_argument("project", nargs="?", default=None, help="Filter by project")
    jobs_p.add_argument("-a", "--active", action="store_true",
                        help="Only show queued/running jobs")

    log_p = sub.add_parser("log", help="Read job output")
    log_p.add_argument("job_id", type=int, nargs="?", default=None,
                       help="Job ID (default: AUTOLEASE_JOB_ID or last job)")
    log_p.add_argument("--stderr", action="store_true", help="Show stderr instead")
    log_p.add_argument("-n", "--tail", type=int, default=None,
                       help="Show last N lines")

    wait_p = sub.add_parser("wait", help="Block until job finishes, print output")
    wait_p.add_argument("job_id", type=int, help="Job ID")
    wait_p.add_argument("--poll", type=float, default=2.0,
                        help="Poll interval in seconds (default: 2)")

    poll_p = sub.add_parser("poll", help="Tail stdout+stderr of a job, refreshing periodically")
    poll_p.add_argument("job_id", type=int, nargs="?", default=None,
                        help="Job ID (default: AUTOLEASE_JOB_ID)")
    poll_p.add_argument("-n", "--tail", type=int, default=None,
                        help="Initial lines to show (default: config poll_tail_lines)")
    poll_p.add_argument("-i", "--interval", type=float, default=None,
                        help="Refresh interval in seconds (default: config poll_interval)")

    cancel_p = sub.add_parser("cancel", help="Cancel queued or running jobs")
    cancel_p.add_argument("job_ids", type=int, nargs="*",
                          help="Job IDs (default: AUTOLEASE_JOB_ID)")

    redo_p = sub.add_parser("redo", help="Re-submit the same command as a previous job")
    redo_p.add_argument("job_id", type=int, nargs="?", default=None,
                        help="Job ID to redo (default: AUTOLEASE_JOB_ID or last job)")
    redo_p.add_argument("--poll", action="store_true",
                        help="Auto-start polling after submit")

    # Sync
    sync_p = sub.add_parser("sync", help="Sync code files to cluster")
    sync_p.add_argument("--dry-run", action="store_true", help="Show what would be synced")

    pull_p = sub.add_parser("pull", help="Pull files from cluster")
    pull_p.add_argument("path", help="Remote subpath to pull (e.g. results/)")

    # Interactive shell on a lease
    shell_p = sub.add_parser("shell", help="Open an interactive shell on a held lease")
    shell_p.add_argument("lease_id", type=int, nargs="?", default=None,
                         help="Lease job ID (default: first running lease)")
    shell_p.add_argument("-g", "--gpu-type", default=None,
                         help="Match a lease with this GPU type")
    shell_p.add_argument("-n", "--num-gpus", type=int, default=1,
                         help="GPUs to request from the lease (default: 1)")
    shell_p.add_argument("-s", "--shell", default=None,
                         help="Shell to run (default: config.shell)")

    # TUI
    sub.add_parser("tui", help="Interactive dashboard")

    # Info
    events_p = sub.add_parser("events", help="Show dispatch/preemption event log")
    events_p.add_argument("-n", "--tail", type=int, default=20,
                          help="Show last N events (default: 20)")
    sub.add_parser("nodes", help="Show cluster GPU nodes")
    parts_p = sub.add_parser("partitions", help="Show partition/QoS map")
    parts_p.add_argument("--refresh", action="store_true",
                         help="Bypass discovery cache (force a fresh scontrol/sinfo round-trip)")
    sub.add_parser("ssh-reset",
                   help="Tear down stale SSH ControlMaster sockets (use after network blips)")

    init_p = sub.add_parser("shell-init",
                            help="Print shell helpers (al-run, al-redo) for sourcing")
    init_p.add_argument("shell", choices=["bash", "zsh", "fish"],
                        help="Target shell")

    recover_p = sub.add_parser("recover",
                               help="Re-check failed jobs to rescue ones whose wrapper is actually still alive")
    recover_p.add_argument("job_id", type=int, nargs="?", default=None,
                           help="Specific job (default: all failed jobs with no exit_code)")

    info_p = sub.add_parser("info",
                            help="One-shot dump of everything autolease knows about a job")
    info_p.add_argument("job_ids", type=int, nargs="*",
                        help="Job IDs (default: AUTOLEASE_JOB_ID)")
    info_p.add_argument("-n", "--tail", type=int, default=20,
                        help="Lines of log to show (default: 20)")
    info_p.add_argument("--no-log", action="store_true",
                        help="Skip the log tail")

    args = p.parse_args()

    # Load config (local only — no SSH)
    cfg = load_config(args.config)
    apply_qos_config(cfg)

    if args.cmd is None:
        from .tui import run_tui
        run_tui(args.config)
        return

    # Only discover partitions for commands that need it
    needs_discovery = {"up", "partitions", "nodes", "pool", "check", "test", "run", "redo"}
    if args.cmd in needs_discovery:
        from .slurm import Slurm, SlurmConfig
        force = (args.cmd == "partitions" and getattr(args, "refresh", False))
        discover_partitions(
            Slurm(SlurmConfig(ssh_host=cfg.ssh_host, shell=cfg.shell)),
            max_age_seconds=cfg.discovery_cache_seconds,
            force=force,
        )

    cmds = {
        "up": cmd_up,
        "down": cmd_down,
        "pool": cmd_pool_status,
        "check": cmd_check,
        "test": cmd_test,
        "renew": cmd_renew,
        "bad-nodes": cmd_bad_nodes,
        "run": cmd_run,
        "status": cmd_status,
        "jobs": cmd_jobs,
        "log": cmd_log,
        "wait": cmd_wait,
        "poll": cmd_poll,
        "cancel": cmd_cancel,
        "redo": cmd_redo,
        "shell": cmd_shell,
        "tui": lambda args: __import__('autolease.tui', fromlist=['run_tui']).run_tui(args.config),
        "sync": cmd_sync,
        "pull": cmd_pull,
        "events": cmd_events,
        "nodes": cmd_nodes,
        "partitions": cmd_partitions,
        "ssh-reset": cmd_ssh_reset,
        "recover": cmd_recover,
        "info": cmd_info,
        "shell-init": cmd_shell_init,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
