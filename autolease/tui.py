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

    def on_mount(self) -> None:
        table = self.query_one("#pool-table", DataTable)
        table.add_columns("Job ID", "Partition", "GPU", "#", "Node", "QoS", "State", "Time Left")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def get_selected_job_id(self) -> int | None:
        table = self.query_one("#pool-table", DataTable)
        if table.cursor_row is not None:
            try:
                return int(table.get_row_at(table.cursor_row)[0])
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
        Binding("l", "view_log", "Log"),
        Binding("e", "view_stderr", "Stderr"),
        Binding("c", "cancel_job", "Cancel"),
        Binding("h", "health_check", "Check"),
    ]

    def __init__(self, config_path: str | None = None):
        super().__init__()
        self._config = load_config(config_path)
        self._pool = Pool(self._config)
        self._queue = JobQueue(self._config)
        self._refresh_timer: Timer | None = None
        self._known_leases: dict[int, str] = {}
        # Discover cluster partitions if not already populated
        if not PARTITION_INFO:
            discover_partitions(self._pool.slurm)
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
        self._refresh_timer = self.set_interval(5.0, self._do_refresh)

    # ── Refresh ──

    def _do_refresh(self) -> None:
        self.run_worker(self._refresh_in_thread, thread=True,
                        exclusive=True, group="refresh")

    def _refresh_in_thread(self) -> None:
        try:
            leases = self._pool.status()
            jobs = self._queue.list_jobs()
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

        # Leases table
        table = self.query_one("#pool-table", DataTable)
        table.clear()
        for l in leases:
            remaining = self._pool.remaining_minutes(l)
            if remaining is not None:
                rem = f"{remaining/60:.1f}h" if remaining >= 60 else f"{remaining:.0f}m"
            else:
                rem = "—"
            table.add_row(
                str(l.job_id), l.partition, l.gpu_type,
                str(l.num_gpus), l.node or "—", l.qos, l.state, rem,
            )

        # Jobs table
        jobs_table = self.query_one("#jobs-table", DataTable)
        jobs_table.clear()
        active = [j for j in jobs if j.state in ("queued", "running")]
        done = [j for j in jobs if j.state not in ("queued", "running")]
        for j in active + done[-10:]:
            state = f"done:{j.exit_code}" if j.state == "done" else j.state
            vram = f"{j.min_vram}G" if j.min_vram else "—"
            cmd = j.command[:40] + ".." if len(j.command) > 42 else j.command
            jobs_table.add_row(
                str(j.id), state, j.project,
                str(j.num_gpus), vram, j.node or "—", cmd,
            )

        # Status bar
        n_leases = len([l for l in leases if l.state == "RUNNING"])
        n_pending = len([l for l in leases if l.state == "PENDING"])
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
        def _do():
            n = self._pool.down()
            def _report():
                self.notify(f"Released {n} lease(s)")
                self._do_refresh()
            self.call_from_thread(_report)
        self.run_worker(_do, thread=True, group="release")

    # ── Job actions ──

    def action_refresh(self) -> None:
        self._do_refresh()

    def action_view_log(self) -> None:
        job_id = self.query_one(JobsPanel).get_selected_job_id()
        if job_id:
            self.query_one(LogPanel).show_log(self._queue, job_id)

    def action_view_stderr(self) -> None:
        job_id = self.query_one(JobsPanel).get_selected_job_id()
        if job_id:
            self.query_one(LogPanel).show_stderr(self._queue, job_id)

    def action_cancel_job(self) -> None:
        job_id = self.query_one(JobsPanel).get_selected_job_id()
        if job_id:
            self._queue.cancel(job_id)
            self.notify(f"Cancelled job {job_id}")
            self._do_refresh()

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
