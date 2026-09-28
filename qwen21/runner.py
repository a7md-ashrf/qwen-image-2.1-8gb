"""Tiny cross-platform process supervisor.

No systemd, no launchd, no Windows services: one `qwen21 start` brings up
ComfyUI, the API and (optionally) the tunnel, and `qwen21 stop` takes them down.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0


@dataclass
class Service:
    name: str
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    url: str | None = None  # readiness probe
    startup_timeout: float = 600.0


class Runner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runtime = root / "runtime"
        self.logs = self.runtime / "logs"
        self.pids = self.runtime / "pids"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.pids.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- state

    def pid_file(self, name: str) -> Path:
        return self.pids / f"{name}.pid"

    def log_file(self, name: str) -> Path:
        return self.logs / f"{name}.log"

    def pid(self, name: str) -> int | None:
        path = self.pid_file(name)
        if not path.is_file():
            return None
        try:
            value = int(path.read_text().strip())
        except ValueError:
            return None
        return value if self.alive(value) else None

    @staticmethod
    def alive(pid: int) -> bool:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
            return str(pid) in (out.stdout or "")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def running(self) -> dict[str, int]:
        found = {}
        for path in sorted(self.pids.glob("*.pid")):
            pid = self.pid(path.stem)
            if pid:
                found[path.stem] = pid
        return found

    # ------------------------------------------------------------- lifecycle

    def start(self, service: Service) -> bool:
        """Start a service and wait for it to answer. False = never came up."""
        existing = self.pid(service.name)
        if existing:
            print(f"  • {service.name} already running (pid {existing})")
            return True
        environment = {**os.environ, **service.env}
        with self.log_file(service.name).open("ab") as log:
            log.write(
                f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{' '.join(service.cmd)}\n".encode()
            )
            log.flush()
            process = subprocess.Popen(
                service.cmd,
                cwd=str(service.cwd),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # Own process group / console group, so stopping the service
                # never signals this CLI process.
                start_new_session=os.name != "nt",
                creationflags=CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        self.pid_file(service.name).write_text(str(process.pid))
        print(f"  ✓ {service.name} (pid {process.pid}) -> {self.log_file(service.name).name}")
        if not service.url:
            return True
        if self.wait_http(service.url, name=service.name, timeout=service.startup_timeout):
            return True
        print(f"  ! last lines of {self.log_file(service.name).name}:")
        for line in self.tail(service.name, 12).splitlines():
            print(f"      {line}")
        return False

    def stop(self, name: str, timeout: float = 20.0) -> bool:
        pid = self.pid(name)
        if not pid:
            self.pid_file(name).unlink(missing_ok=True)
            return False
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                os.kill(pid, signal.SIGTERM)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive(pid):
                break
            time.sleep(0.2)
        else:
            if os.name != "nt":
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        self.pid_file(name).unlink(missing_ok=True)
        return True

    def stop_all(self) -> None:
        for name in ("tunnel", "api", "comfy"):
            if self.stop(name):
                print(f"  ✓ stopped {name}")

    # -------------------------------------------------------------- helpers

    @staticmethod
    def wait_http(url: str, timeout: float = 600.0, name: str = "service") -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=3):
                    print(f"  ✓ {name} is answering on {url}")
                    return True
            except (TimeoutError, urllib.error.URLError, OSError):
                time.sleep(1.0)
        print(f"  ! {name} did not answer on {url} after {timeout:.0f}s")
        return False

    def tail(self, name: str, lines: int = 40) -> str:
        path = self.log_file(name)
        if not path.is_file():
            return f"(no log for {name})"
        return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])
