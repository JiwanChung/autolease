"""CLI entry point for autolease."""

import argparse
import json
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
    )
    print(job.id)


def cmd_status(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    job = q.get(args.job_id)
    if job is None:
        print(f"unknown", file=sys.stderr)
        sys.exit(1)
    if args.json:
        from dataclasses import asdict
        print(json.dumps(asdict(job)))
    else:
        if job.state == "done":
            print(f"done:{job.exit_code}")
        else:
            print(job.state)


def cmd_jobs(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    jobs = q.list_jobs(project=args.project, active_only=args.active)
    if not jobs:
        print("No jobs.")
        return
    print(f"{'ID':>5}  {'STATE':<10}  {'PROJECT':<20}  {'GPU':>4}  {'VRAM':>4}  {'NODE':<10}  {'CMD':<30}")
    print("-" * 95)
    for j in jobs:
        state = f"done:{j.exit_code}" if j.state == "done" else j.state
        cmd_short = j.command[:28] + ".." if len(j.command) > 30 else j.command
        vram = f"{j.min_vram}G" if j.min_vram else "—"
        print(f"{j.id:>5}  {state:<10}  {j.project:<20}  {j.num_gpus:>4}  {vram:>4}  {j.node or '—':<10}  {cmd_short:<30}")


def cmd_log(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    stream = "stderr" if args.stderr else "stdout"
    out = q.read_log(args.job_id, stream=stream, tail=args.tail)
    if out:
        print(out, end="")
    else:
        job = q.get(args.job_id)
        if job is None:
            print(f"Job {args.job_id} not found.", file=sys.stderr)
        elif job.state == "queued":
            print(f"Job {args.job_id} is queued (no output yet).", file=sys.stderr)


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


def cmd_cancel(args):
    cfg = load_config(args.config)
    q = JobQueue(cfg)
    ok = q.cancel(args.job_id)
    if ok:
        print(f"Cancelled job {args.job_id}.")
    else:
        print(f"Job {args.job_id} not found or already finished.", file=sys.stderr)
        sys.exit(1)


# ── Info commands ──

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

    check_p = sub.add_parser("check", help="Health-check running leases")
    check_p.add_argument("--replace", action="store_true",
                         help="Auto-replace bad leases")

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
    run_p.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    status_p = sub.add_parser("status", help="Get job state (one word)")
    status_p.add_argument("job_id", type=int, help="Job ID")
    status_p.add_argument("--json", action="store_true", help="Full JSON output")

    jobs_p = sub.add_parser("jobs", help="List jobs")
    jobs_p.add_argument("project", nargs="?", default=None, help="Filter by project")
    jobs_p.add_argument("-a", "--active", action="store_true",
                        help="Only show queued/running jobs")

    log_p = sub.add_parser("log", help="Read job output")
    log_p.add_argument("job_id", type=int, help="Job ID")
    log_p.add_argument("--stderr", action="store_true", help="Show stderr instead")
    log_p.add_argument("-n", "--tail", type=int, default=None,
                       help="Show last N lines")

    wait_p = sub.add_parser("wait", help="Block until job finishes, print output")
    wait_p.add_argument("job_id", type=int, help="Job ID")
    wait_p.add_argument("--poll", type=float, default=2.0,
                        help="Poll interval in seconds (default: 2)")

    cancel_p = sub.add_parser("cancel", help="Cancel a queued or running job")
    cancel_p.add_argument("job_id", type=int, help="Job ID")

    # TUI
    sub.add_parser("tui", help="Interactive dashboard")

    # Info
    sub.add_parser("nodes", help="Show cluster GPU nodes")
    sub.add_parser("partitions", help="Show partition/QoS map")

    args = p.parse_args()

    # Load config + discover cluster
    cfg = load_config(args.config)
    from .slurm import Slurm, SlurmConfig
    discover_partitions(Slurm(SlurmConfig(ssh_host=cfg.ssh_host, shell=cfg.shell)))
    apply_qos_config(cfg)

    if args.cmd is None:
        from .tui import run_tui
        run_tui(args.config)
        return

    cmds = {
        "up": cmd_up,
        "down": cmd_down,
        "pool": cmd_pool_status,
        "check": cmd_check,
        "renew": cmd_renew,
        "bad-nodes": cmd_bad_nodes,
        "run": cmd_run,
        "status": cmd_status,
        "jobs": cmd_jobs,
        "log": cmd_log,
        "wait": cmd_wait,
        "cancel": cmd_cancel,
        "tui": lambda args: __import__('autolease.tui', fromlist=['run_tui']).run_tui(args.config),
        "nodes": cmd_nodes,
        "partitions": cmd_partitions,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
