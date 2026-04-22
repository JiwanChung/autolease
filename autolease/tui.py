"""Interactive TUI for autolease."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable, Footer, Header, Static, Log, Input, Button, Label, Rule,
)
from textual.timer import Timer
from textual.worker import Worker

from .config import (
    load_config, LeaseSpec, PARTITION_INFO, GPU_VRAM, pick_qos,
    discover_partitions, apply_qos_config,
)
from .pool import Pool
from .queue import JobQueue


# ── Modal: Confirmation ──

class ConfirmScreen(ModalScreen[bool]):
    """Simple yes/no confirmation dialog."""

    CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 60;
        height: auto;
        max-height: 12;
        background: $surface;
        border: round $error;
        padding: 1 2;
    }
    #confirm-msg {
        width: 100%;
        height: auto;
        content-align: center middle;
        margin-bottom: 1;
    }
    #confirm-actions {
        height: auto;
        align: center middle;
    }
    #confirm-actions Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, message: str):
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(self._message, id="confirm-msg")
            with Horizontal(id="confirm-actions"):
                yield Button("Yes (y)", variant="error", id="btn-yes")
                yield Button("No (n)", variant="default", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


# ── Modal: Acquire Lease ──

class AcquireLeaseScreen(ModalScreen[LeaseSpec | None]):

    CSS = """
    AcquireLeaseScreen {
        align: center middle;
    }
    #acquire-dialog {
        width: 94;
        height: 34;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    #acquire-title {
        text-style: bold;
        color: $text;
        width: 100%;
        content-align: center middle;
        height: 1;
        margin-bottom: 1;
    }
    #avail-table {
        height: 1fr;
    }
    #acquire-form {
        height: auto;
        margin-top: 1;
    }
    #acquire-form Label {
        width: 8;
        height: 3;
        content-align: left middle;
        margin-right: 1;
    }
    #acquire-form Input {
        width: 1fr;
    }
    #acquire-actions {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    #acquire-actions Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Acquire"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="acquire-dialog"):
            yield Static("Acquire Lease", id="acquire-title")
            yield DataTable(id="avail-table")
            with Horizontal(id="acquire-form"):
                yield Label("GPUs")
                yield Input("1", id="gpu-count")
                yield Label("Time")
                yield Input("", id="time-limit", placeholder="max")
            with Horizontal(id="acquire-actions"):
                yield Button("Acquire", variant="primary", id="btn-acquire")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def on_mount(self) -> None:
        table = self.query_one("#avail-table", DataTable)
        table.add_columns("Node", "Partition", "GPU", "VRAM", "Free", "Total", "State")
        table.cursor_type = "row"
        table.zebra_stripes = True

        nodes = getattr(self.app, "_acquire_nodes", [])
        for n in nodes:
            vram = GPU_VRAM.get(n["gpu_type"], "?")
            vram_str = f"{vram}GB" if isinstance(vram, int) else vram
            table.add_row(
                n["node"], n["partition"], n["gpu_type"], vram_str,
                str(n["free"]), str(n["total"]), n["state"],
            )

    def _get_selected_partition(self) -> str | None:
        table = self.query_one("#avail-table", DataTable)
        if table.cursor_row is not None:
            try:
                row = table.get_row_at(table.cursor_row)
                return row[1]
            except (IndexError, ValueError):
                pass
        return None

    def _build_spec(self) -> LeaseSpec | None:
        partition = self._get_selected_partition()
        if not partition:
            return None
        num_gpus = int(self.query_one("#gpu-count", Input).value or "1")
        time_val = self.query_one("#time-limit", Input).value.strip()
        time_limit = time_val if time_val else None
        usage = getattr(self.app, "_qos_usage", {})
        qos = pick_qos(partition, num_gpus, usage)
        return LeaseSpec(
            partition=partition, qos=qos,
            num_gpus=num_gpus, time=time_limit,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-acquire":
            self.dismiss(self._build_spec())
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def action_confirm(self) -> None:
        self.dismiss(self._build_spec())

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Panels ──

class PoolPanel(Static):

    def compose(self) -> ComposeResult:
        yield DataTable(id="pool-table")

    # Column order: GPU/#/Node/QoS/State/In Use/Time Left/Partition/Job ID
    _JOB_ID_COL = 8

    def on_mount(self) -> None:
        table = self.query_one("#pool-table", DataTable)
        table.add_columns("GPU", "#", "Node", "QoS", "State", "In Use", "Time Left", "Partition", "Job ID")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def get_selected_job_id(self) -> int | None:
        table = self.query_one("#pool-table", DataTable)
        if table.cursor_row is not None:
            try:
                return int(table.get_row_at(table.cursor_row)[self._JOB_ID_COL])
            except (IndexError, ValueError):
                pass
        return None


class JobsPanel(Static):

    def compose(self) -> ComposeResult:
        yield DataTable(id="jobs-table")

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns("ID", "State", "Project", "GPU", "VRAM", "Node", "Command")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def get_selected_job_id(self) -> int | None:
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is not None:
            try:
                return int(table.get_row_at(table.cursor_row)[0])
            except (IndexError, ValueError):
                pass
        return None


class LogPanel(Static):

    def compose(self) -> ComposeResult:
        yield Static("[dim]select a job, press L[/dim]", id="log-header")
        yield Log(id="log-output", auto_scroll=True)

    def show_log(self, queue: JobQueue, job_id: int) -> None:
        header = self.query_one("#log-header", Static)
        log_widget = self.query_one("#log-output", Log)
        log_widget.clear()
        header.update(f"[bold]job {job_id}[/bold] stdout")
        out = queue.read_log(job_id, stream="stdout")
        log_widget.write(out if out else "(no output yet)\n")

    def show_stderr(self, queue: JobQueue, job_id: int) -> None:
        header = self.query_one("#log-header", Static)
        log_widget = self.query_one("#log-output", Log)
        log_widget.clear()
        header.update(f"[bold]job {job_id}[/bold] stderr")
        err = queue.read_log(job_id, stream="stderr")
        log_widget.write(err if err else "(no stderr)\n")


# ── App ──

class AutoleaseApp(App):

    TITLE = "autolease"
    SUB_TITLE = "GPU pool manager"

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr 1fr;
    }
    PoolPanel {
        border: round $primary 50%;
        border-title-color: $primary;
        padding: 0 1;
        column-span: 1;
    }
    JobsPanel {
        border: round $secondary 50%;
        border-title-color: $secondary;
        padding: 0 1;
        column-span: 1;
    }
    LogPanel {
        border: round $accent 50%;
        border-title-color: $accent;
        padding: 0 1;
        column-span: 2;
    }
    #log-header {
        height: 1;
        color: $text-muted;
    }
    #log-output {
        height: 1fr;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        background: $primary-background;
        color: $text-muted;
        padding: 0 1;
        column-span: 2;
    }
    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("a", "acquire_lease", "+Lease"),
        Binding("d", "release_lease", "-Lease"),
        Binding("shift+d", "release_all", "-All"),
        Binding("e", "toggle_log", "Log/Err"),
        Binding("c", "cancel", "Cancel"),
        Binding("shift+h", "health_check", "Check"),
        # vim-style navigation (not shown in footer)
        Binding("h", "focus_left", show=False),
        Binding("l", "focus_right", show=False),
        Binding("k", "cursor_up", show=False),
        Binding("j", "cursor_down", show=False),
    ]

    def __init__(self, config_path: str | None = None):
        super().__init__()
        self._config = load_config(config_path)
        self._pool = Pool(self._config)
        self._queue = JobQueue(self._config)
        self._refresh_timer: Timer | None = None
        self._known_leases: dict[int, str] = {}
        self._discovered = False
        self._log_stream: str = "stdout"  # current log stream shown
        self._log_job_id: int | None = None  # job whose log is shown
        self._running_jobs: dict[int, int] = {}  # lease_job_id -> job_id for "in use" display
        self._shutting_down = False
        apply_qos_config(self._config)

    def compose(self) -> ComposeResult:
        yield Header()

        pool_panel = PoolPanel()
        pool_panel.border_title = "leases"
        yield pool_panel

        jobs_panel = JobsPanel()
        jobs_panel.border_title = "jobs"
        yield jobs_panel

        log_panel = LogPanel()
        log_panel.border_title = "output"
        yield log_panel

        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._do_refresh()
        self._refresh_timer = self.set_interval(30.0, self._do_refresh)

    def _selected_job_id(self) -> int | None:
        """Get the currently selected job ID from the jobs table."""
        return self.query_one(JobsPanel).get_selected_job_id()

    def _refresh_log_for_selected(self) -> None:
        """Load log for whatever job is currently selected."""
        job_id = self._selected_job_id()
        if job_id is not None:
            self._log_job_id = job_id
            self._load_log_async(job_id, self._log_stream)

    def on_data_table_cursor_moved(self, event) -> None:
        """Auto-load log when cursor moves in the jobs table (arrows, mouse, any method)."""
        try:
            table = event.control if hasattr(event, "control") else event.data_table
        except Exception:
            return
        if table.id == "jobs-table":
            self._refresh_log_for_selected()

    # ── Refresh ──

    def _do_refresh(self) -> None:
        self.run_worker(self._refresh_in_thread, thread=True,
                        exclusive=True, group="refresh")

    def _refresh_in_thread(self) -> None:
        try:
            if self._shutting_down:
                return
            if not self._discovered:
                discover_partitions(self._pool.slurm)
                self._discovered = True
            if self._shutting_down:
                return
            # Single squeue for leases
            leases = self._pool.status()
            if self._shutting_down:
                return
            # Refresh each running job (1 SSH per job now)
            for j in self._queue._all_jobs():
                if self._shutting_down:
                    return
                if j.state == "running":
                    self._queue._refresh_running(j)
            if self._shutting_down:
                return
            # Dispatch pending jobs — reuse the refresh work we just did.
            # No-op if there are no queued jobs (zero SSH).
            self._queue.dispatch(leases=leases, skip_running_refresh=True)
            jobs = self._queue._all_jobs()
            bad = self._pool.bad_nodes()
            self.call_from_thread(self._apply_refresh, leases, jobs, bad)
        except Exception:
            pass

    def _apply_refresh(self, leases, jobs, bad) -> None:
        # Detect lost leases
        current_ids = {l.job_id for l in leases}
        for old_id, old_node in list(self._known_leases.items()):
            if old_id not in current_ids:
                self.notify(
                    f"Lease {old_id} on {old_node} lost (preempted/expired)",
                    severity="error", timeout=15,
                )
                self.bell()
        self._known_leases = {
            l.job_id: (l.node or "?") for l in leases if l.state == "RUNNING"
        }

        # Build lease->job mapping for "in use" column
        self._running_jobs = {}
        for j in jobs:
            if j.state == "running" and j.lease_job_id:
                self._running_jobs[j.lease_job_id] = j.id

        # Leases table
        table = self.query_one("#pool-table", DataTable)
        table.clear()
        for l in leases:
            remaining = self._pool.remaining_minutes(l)
            if remaining is not None:
                rem = f"{remaining/60:.1f}h" if remaining >= 60 else f"{remaining:.0f}m"
            else:
                rem = "—"
            in_use = str(self._running_jobs[l.job_id]) if l.job_id in self._running_jobs else "—"
            table.add_row(
                l.gpu_type, str(l.num_gpus), l.node or "—", l.qos, l.state,
                in_use, rem, l.partition, str(l.job_id),
            )

        # Jobs table
        jobs_table = self.query_one("#jobs-table", DataTable)
        jobs_table.clear()
        active = [j for j in jobs if j.state in ("queued", "running")]
        done = [j for j in jobs if j.state not in ("queued", "running")]
        # Build job->gpu_type lookup from leases
        lease_gpu = {l.job_id: l.gpu_type for l in leases}
        for j in active + done[-10:]:
            state = f"done:{j.exit_code}" if j.state == "done" else j.state
            # Show actual GPU VRAM if running on a lease, else show min_vram requirement
            gpu_type = lease_gpu.get(j.lease_job_id) if j.lease_job_id else None
            if gpu_type:
                vram_val = GPU_VRAM.get(gpu_type, 0)
                vram = f"{vram_val}G" if vram_val else "—"
            elif j.min_vram:
                vram = f"≥{j.min_vram}G"
            else:
                vram = "—"
            cmd = j.command[:40] + ".." if len(j.command) > 42 else j.command
            jobs_table.add_row(
                str(j.id), state, j.project,
                str(j.num_gpus), vram, j.node or "—", cmd,
            )

        # Lease panel aggregate: per-QoS usage
        from .config import QOS_GPU_LIMITS
        qos_used: dict[str, int] = {}
        for l in leases:
            if l.state in ("RUNNING", "PENDING"):
                qos_used[l.qos] = qos_used.get(l.qos, 0) + l.num_gpus
        lease_parts = []
        for qos in sorted(qos_used.keys()):
            limit = QOS_GPU_LIMITS.get(qos, 0)
            used = qos_used[qos]
            if limit:
                lease_parts.append(f"{qos}:{used}/{limit}")
            else:
                lease_parts.append(f"{qos}:{used}")
        n_running = len([l for l in leases if l.state == "RUNNING"])
        n_pending_l = len([l for l in leases if l.state == "PENDING"])
        lease_title = f"leases ({n_running} running"
        if n_pending_l:
            lease_title += f", {n_pending_l} pending"
        lease_title += ")"
        if lease_parts:
            lease_title += "  " + "  ".join(lease_parts)
        self.query_one(PoolPanel).border_title = lease_title

        # Jobs panel aggregate: running/queued counts, GPUs used
        n_running_j = len([j for j in active if j.state == "running"])
        n_queued_j = len([j for j in active if j.state == "queued"])
        gpus_used = sum(j.num_gpus for j in active if j.state == "running")
        job_title = f"jobs ({n_running_j} running"
        if n_queued_j:
            job_title += f", {n_queued_j} queued"
        job_title += f", {gpus_used} GPU{'s' if gpus_used != 1 else ''})"
        self.query_one(JobsPanel).border_title = job_title

        # Status bar
        n_leases = n_running
        n_pending = n_pending_l
        n_active = len(active)
        parts = [f"{n_leases} lease{'s' if n_leases != 1 else ''}"]
        if n_pending:
            parts.append(f"{n_pending} pending")
        parts.append(f"{n_active} job{'s' if n_active != 1 else ''} active")
        if bad:
            parts.append(f"bad: {','.join(bad)}")
        self.query_one("#status-bar", Static).update("  ".join(parts))

    # ── Lease actions ──

    def action_acquire_lease(self) -> None:
        self.notify("Loading availability...")
        def _fetch() -> tuple[list[dict], dict[str, int]]:
            nodes = self._pool.slurm.node_gpu_availability()
            usage = self._pool.slurm.gpu_usage_by_qos()
            return nodes, usage
        self.run_worker(_fetch, thread=True, group="acquire-fetch",
                        name="acquire-fetch")

    def on_worker_state_changed(self, event) -> None:
        if (event.worker.name == "acquire-fetch"
                and event.worker.state.name == "SUCCESS"):
            nodes, usage = event.worker.result
            self._acquire_nodes = nodes
            self._qos_usage = usage
            self.push_screen(AcquireLeaseScreen(), self._on_acquire_result)

    def _on_acquire_result(self, spec: LeaseSpec | None) -> None:
        if spec is None:
            return
        self.notify(f"Acquiring {spec.partition} x{spec.num_gpus} ({spec.qos})...")

        def _do():
            try:
                lease = self._pool.up(spec)
                ok = self._pool.wait_and_check(lease, poll_interval=3, max_wait=120)
                def _report():
                    if ok:
                        self.notify(f"Lease {lease.job_id} on {lease.node}: OK")
                    else:
                        self.notify(f"Lease {lease.job_id}: health check failed",
                                    severity="error")
                    self._do_refresh()
                self.call_from_thread(_report)
            except RuntimeError as e:
                def _err():
                    self.notify(f"Failed: {e}", severity="error")
                self.call_from_thread(_err)

        self.run_worker(_do, thread=True, group="acquire")

    def action_release_lease(self) -> None:
        job_id = self.query_one(PoolPanel).get_selected_job_id()
        if job_id is None:
            self.notify("Select a lease first", severity="warning")
            return
        in_use = job_id in self._running_jobs
        msg = f"Release lease {job_id}?"
        if in_use:
            msg = f"Release lease {job_id}? (job {self._running_jobs[job_id]} running on it!)"
        self.push_screen(ConfirmScreen(msg), lambda ok: self._do_release_lease(job_id) if ok else None)

    def _do_release_lease(self, job_id: int) -> None:
        def _do():
            try:
                self._pool.release(job_id)
                def _report():
                    self.notify(f"Released lease {job_id}")
                    self._do_refresh()
                self.call_from_thread(_report)
            except RuntimeError as e:
                def _err():
                    self.notify(f"Failed: {e}", severity="error")
                self.call_from_thread(_err)
        self.run_worker(_do, thread=True, group="release")

    def action_release_all(self) -> None:
        n_jobs = len(self._running_jobs)
        msg = "Release ALL leases?"
        if n_jobs:
            msg = f"Release ALL leases? ({n_jobs} job(s) still running!)"
        self.push_screen(ConfirmScreen(msg), lambda ok: self._do_release_all() if ok else None)

    def _do_release_all(self) -> None:
        def _do():
            n = self._pool.down()
            def _report():
                self.notify(f"Released {n} lease(s)")
                self._do_refresh()
            self.call_from_thread(_report)
        self.run_worker(_do, thread=True, group="release")

    # ── Navigation ──

    def _focused_table(self) -> DataTable | None:
        """Find the DataTable that currently has focus."""
        focused = self.focused
        if isinstance(focused, DataTable):
            return focused
        return None

    def action_cursor_up(self) -> None:
        table = self._focused_table()
        if table:
            table.action_cursor_up()

    def action_cursor_down(self) -> None:
        table = self._focused_table()
        if table:
            table.action_cursor_down()

    def action_focus_left(self) -> None:
        self.query_one("#pool-table", DataTable).focus()

    def action_focus_right(self) -> None:
        self.query_one("#jobs-table", DataTable).focus()
        self._refresh_log_for_selected()

    # ── Job actions ──

    def action_refresh(self) -> None:
        self._do_refresh()

    def _load_log_async(self, job_id: int, stream: str) -> None:
        """Load log in a background thread to avoid blocking the UI."""
        def _fetch():
            if self._shutting_down:
                return
            text = self._queue.read_log(job_id, stream=stream)
            def _show():
                # Only update if we're still looking at this job/stream
                if self._log_job_id != job_id:
                    return
                log_panel = self.query_one(LogPanel)
                header = log_panel.query_one("#log-header", Static)
                log_widget = log_panel.query_one("#log-output", Log)
                log_widget.clear()
                header.update(f"[bold]job {job_id}[/bold] {stream}")
                log_widget.write(text if text else f"(no {stream})\n")
            self.call_from_thread(_show)
        self.run_worker(_fetch, thread=True, exclusive=True, group="log")

    def action_toggle_log(self) -> None:
        """Toggle between stdout and stderr for the current job."""
        job_id = self.query_one(JobsPanel).get_selected_job_id()
        if not job_id:
            return
        # If viewing a different job, start with stdout
        if job_id != self._log_job_id:
            self._log_stream = "stdout"
            self._log_job_id = job_id
        else:
            # Toggle
            self._log_stream = "stderr" if self._log_stream == "stdout" else "stdout"
        self._load_log_async(job_id, self._log_stream)

    def action_cancel(self) -> None:
        """Cancel selected lease or job depending on which panel has focus."""
        focused = self.focused
        # Walk up to find which panel we're in
        widget = focused
        while widget is not None:
            if isinstance(widget, PoolPanel):
                self._cancel_selected_lease()
                return
            if isinstance(widget, JobsPanel):
                self._cancel_selected_job()
                return
            widget = widget.parent
        # Default: try job panel
        self._cancel_selected_job()

    def _cancel_selected_job(self) -> None:
        job_id = self.query_one(JobsPanel).get_selected_job_id()
        if job_id is None:
            self.notify("Select a job first", severity="warning")
            return
        self.push_screen(
            ConfirmScreen(f"Cancel job {job_id}?"),
            lambda ok: self._do_cancel_job(job_id) if ok else None,
        )

    def _do_cancel_job(self, job_id: int) -> None:
        self._queue.cancel(job_id)
        self.notify(f"Cancelled job {job_id}")
        self._do_refresh()

    def _cancel_selected_lease(self) -> None:
        job_id = self.query_one(PoolPanel).get_selected_job_id()
        if job_id is None:
            self.notify("Select a lease first", severity="warning")
            return
        in_use = job_id in self._running_jobs
        msg = f"Release lease {job_id}?"
        if in_use:
            msg = f"Release lease {job_id}? (job {self._running_jobs[job_id]} running on it!)"
        self.push_screen(
            ConfirmScreen(msg),
            lambda ok: self._do_release_lease(job_id) if ok else None,
        )

    def action_quit(self) -> None:
        self._shutting_down = True
        if self._refresh_timer:
            self._refresh_timer.stop()
        for worker in self.workers:
            worker.cancel()
        self.exit()

    def action_health_check(self) -> None:
        def _do():
            actions = self._pool.check_and_replace()
            def _notify():
                for a in actions:
                    if a["action"] == "ok":
                        self.notify(f"Job {a['job_id']}: OK")
                    elif a["action"] == "bad":
                        self.notify(f"Job {a['job_id']} on {a.get('node')}: FAILED",
                                    severity="error")
                    elif a["action"] == "replacement":
                        self.notify(f"Replacement {a['job_id']} submitted",
                                    severity="warning")
                self._do_refresh()
            self.call_from_thread(_notify)
        self.run_worker(_do, thread=True, group="health")


def run_tui(config_path: str | None = None):
    app = AutoleaseApp(config_path=config_path)
    app.run()
