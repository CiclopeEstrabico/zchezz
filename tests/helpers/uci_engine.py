"""Small reusable UCI subprocess driver used by tests and tools."""
from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from pathlib import Path


class UCIError(RuntimeError):
    """Report an engine protocol or lifetime error."""


class UCIEngine:
    def __init__(self, executable: Path | str, *, cwd: Path | str | None = None):
        self.executable = Path(executable)
        self.cwd = Path(cwd) if cwd else self.executable.parent
        self.proc: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[str] = queue.Queue()
        self._stderr: list[str] = []
        self._stdout: list[str] = []
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> "UCIEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        self.proc = subprocess.Popen(
            [str(self.executable)], cwd=str(self.cwd), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        self._threads = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            line = raw.rstrip("\r\n")
            self._stdout.append(line)
            self._queue.put(line)

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in self.proc.stderr:
            self._stderr.append(raw.rstrip("\r\n"))

    @property
    def stdout(self) -> tuple[str, ...]:
        return tuple(self._stdout)

    @property
    def stderr(self) -> tuple[str, ...]:
        return tuple(self._stderr)

    def send(self, command: str) -> None:
        if not self.proc or self.proc.poll() is not None:
            raise UCIError("engine is not running")
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_until(self, pattern: str, *, timeout: float = 10.0) -> list[str]:
        regex = re.compile(pattern)
        deadline = time.monotonic() + timeout
        lines: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self._queue.get(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                if self.proc and self.proc.poll() is not None:
                    break
                continue
            lines.append(line)
            if regex.search(line):
                return lines
        raise UCIError(f"timeout waiting for {pattern!r}; last lines: {lines[-8:]}")

    def handshake(self, *, timeout: float = 5.0) -> list[str]:
        self.send("uci")
        lines = self.read_until(r"^uciok$", timeout=timeout)
        self.send("isready")
        return lines + self.read_until(r"^readyok$", timeout=timeout)

    def setoption(self, name: str, value: str | int | bool) -> None:
        if isinstance(value, bool):
            value = "true" if value else "false"
        self.send(f"setoption name {name} value {value}")

    def search(self, position: str, go: str, *, timeout: float = 15.0) -> list[str]:
        self.send(position)
        self.send(go)
        return self.read_until(r"^bestmove\b", timeout=timeout)

    def stop(self, *, timeout: float = 10.0) -> list[str]:
        self.send("stop")
        return self.read_until(r"^bestmove\b", timeout=timeout)

    def close(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=3)
        self.proc = None
