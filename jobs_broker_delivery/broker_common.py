#!/usr/bin/env python3

from __future__ import annotations

import datetime as _dt
import errno
import fcntl
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

STATUS_BASENAME = "status.json"
STATUS_LOG_BASENAME = "status.log"

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
}

BROKER_GENERATED_JOB_FILE_RE = re.compile(
    r"^jobs/[^/]+/"
    r"(?:status(?:\.conflict-[^/]+)*\.(?:json|log)|slurm-[^/]+(?:\.conflict-[^/]+)*\.(?:out|err))$"
)

TRACKED_JOB_PATH_RE = re.compile(r"^jobs/[^/]+(?:/.*)?$")

BROKER_LOCAL_FILE_RE = re.compile(
    r"^(?:"
    r"\.gsync_broker\.lock|"
    r"\.gsync_broker_state\.json|"
    r"\.gsync_worker\.lock|"
    r"\.gsync_worker_state\.json|"
    r"\.jobs_broker\.lock|"
    r"\.jobs_broker_state\.json|"
    r"\.slurm_worker\.lock|"
    r"\.slurm_worker_state\.json|"
    r"\.broker(?:/.*)?"
    r")$"
)


class CmdError(RuntimeError):
    pass


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    ts = _dt.datetime.now().strftime("%F %T")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    capture: bool = True,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    if cwd is None:
        cwd = Path.cwd()
    disp = " ".join(shlex.quote(c) for c in cmd)
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=capture,
            env=env,
        )
    except FileNotFoundError as e:
        raise CmdError(f"Command not found: {cmd[0]}") from e

    if check and cp.returncode != 0:
        out = (cp.stdout or "").strip()
        err = (cp.stderr or "").strip()
        raise CmdError(
            f"Command failed (rc={cp.returncode}): {disp}\n"
            f"cwd={cwd}\n"
            f"stdout={out}\n"
            f"stderr={err}"
        )
    return cp


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def acquire_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN):
            raise RuntimeError(f"Another broker instance is running (lock: {lock_path}).") from e
        raise
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()} started_at={now_iso()}\n".encode("utf-8"))
    os.lseek(fd, 0, os.SEEK_SET)
    return fd


def is_terminal(state: str) -> bool:
    s = state.strip().upper()
    s0 = s.split()[0] if s else ""
    return s0 in TERMINAL_STATES


def normalize_state(state: str) -> str:
    s = state.strip()
    if not s:
        return s
    return s.split()[0].upper()


def is_tracked_job_path(path: str) -> bool:
    return bool(TRACKED_JOB_PATH_RE.match(path))


def is_broker_generated_job_path(path: str) -> bool:
    return bool(BROKER_GENERATED_JOB_FILE_RE.match(path))


def job_id_from_dir(job_dir: Path) -> str:
    return job_dir.name


def status_path(job_dir: Path) -> Path:
    return job_dir / STATUS_BASENAME


def status_log_path(job_dir: Path) -> Path:
    return job_dir / STATUS_LOG_BASENAME


def read_status(job_dir: Path) -> Optional[dict]:
    p = status_path(job_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_status(job_dir: Path, data: dict) -> bool:
    p = status_path(job_dir)
    old = None
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            old = None
    if old == data:
        return False
    atomic_write_json(p, data)
    return True


def replace_status_log(job_dir: Path, line: str) -> None:
    p = status_log_path(job_dir)
    ts = _dt.datetime.now().strftime("%F %T")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


def append_status_log(job_dir: Path, line: str) -> None:
    p = status_log_path(job_dir)
    ts = _dt.datetime.now().strftime("%F %T")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


def clear_preexisting_managed_files(job_dir: Path) -> List[str]:
    removed: List[str] = []
    job_id = job_id_from_dir(job_dir)
    if not job_dir.exists():
        return removed
    for child in job_dir.iterdir():
        rel = f"jobs/{job_id}/{child.name}"
        if child.is_file() and is_broker_generated_job_path(rel):
            child.unlink()
            removed.append(child.name)
    return sorted(removed)
