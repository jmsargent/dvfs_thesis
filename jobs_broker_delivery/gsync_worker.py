#!/usr/bin/env python3
"""
gsync_worker.py — Root-side Slurm worker.

This process does not use Git. It only:
  - consumes submit/cancel requests from .broker/requests/
  - runs sbatch / squeue / sacct
  - writes result files into .broker/results/
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import pwd
import time
from pathlib import Path
from typing import Dict, List, Optional

from broker_common import (
    CmdError,
    acquire_lock,
    atomic_write_json,
    is_terminal,
    job_id_from_dir,
    log,
    normalize_state,
    now_iso,
    read_status,
    run_cmd,
)

SBATCH_JOBID_RE = __import__("re").compile(r"Submitted batch job\s+(\d+)")


@dataclasses.dataclass
class Config:
    repo_dir: Path
    jobs_dir: Path
    requests_dir: Path
    results_dir: Path
    ready_name: str
    sbatch_name: str
    request_interval_s: int
    status_interval_s: int
    lock_file: Path
    state_file: Path
    submit_as_user: Optional[str]
    job_name_max: int = 120


@dataclasses.dataclass
class State:
    active_jobs: Dict[str, dict]

    @staticmethod
    def empty() -> "State":
        return State(active_jobs={})

    def to_json(self) -> dict:
        return {"active_jobs": self.active_jobs}

    @staticmethod
    def from_json(data: dict) -> "State":
        active: Dict[str, dict] = {}
        for job_id, raw in dict(data.get("active_jobs", {})).items():
            rec = dict(raw)
            if "slurm_job_id" in rec:
                rec["slurm_job_id"] = int(rec["slurm_job_id"])
            active[job_id] = rec
        return State(active_jobs=active)


def load_state(cfg: Config) -> State:
    if cfg.state_file.exists():
        try:
            return State.from_json(json.loads(cfg.state_file.read_text(encoding="utf-8")))
        except Exception as e:
            log(f"[WORKER][WARN] failed to load state {cfg.state_file}: {e}; starting fresh")
    return State.empty()


def save_state(cfg: Config, st: State) -> None:
    atomic_write_json(cfg.state_file, st.to_json())


def detect_submit_as_user(repo_dir: Path) -> Optional[str]:
    if os.geteuid() != 0:
        return None
    try:
        owner = pwd.getpwuid(repo_dir.stat().st_uid).pw_name
    except KeyError:
        return None
    if owner == "root":
        return None
    return owner


def list_ready_jobs(cfg: Config) -> List[Path]:
    ready_files = glob.glob(str(cfg.jobs_dir / "*" / cfg.ready_name))
    jobs = [Path(p).parent for p in ready_files if Path(p).parent.is_dir()]
    jobs.sort()
    return jobs


def list_request_paths(cfg: Config) -> List[Path]:
    submit_paths = sorted(p for p in cfg.requests_dir.glob("*.submit.json") if p.is_file())
    cancel_paths = sorted(p for p in cfg.requests_dir.glob("*.cancel.json") if p.is_file())
    other_paths = sorted(
        p
        for p in cfg.requests_dir.glob("*.json")
        if p.is_file() and p not in submit_paths and p not in cancel_paths
    )
    return submit_paths + cancel_paths + other_paths


def emit_result(cfg: Config, kind: str, job_id: str, payload: dict) -> None:
    stamp = __import__("datetime").datetime.now().strftime("%Y%m%dT%H%M%S%f")
    path = cfg.results_dir / f"{stamp}-{kind}-{job_id}.json"
    atomic_write_json(path, {"type": kind, **payload})


def sbatch_submit(cfg: Config, job_dir: Path) -> int:
    sbatch_script = job_dir / cfg.sbatch_name
    if not sbatch_script.exists():
        raise RuntimeError(f"Missing sbatch script: {sbatch_script}")
    job_name = job_id_from_dir(job_dir)[: cfg.job_name_max]
    out_path = (job_dir / "slurm-%j.out").resolve()
    err_path = (job_dir / "slurm-%j.err").resolve()
    cp = run_cmd(
        [
            "sbatch",
            *([f"--uid={cfg.submit_as_user}"] if cfg.submit_as_user else []),
            f"--job-name={job_name}",
            f"--output={str(out_path)}",
            f"--error={str(err_path)}",
            str(sbatch_script),
        ],
        cwd=cfg.repo_dir,
        capture=True,
        check=True,
    )
    match = SBATCH_JOBID_RE.search((cp.stdout or "") + "\n" + (cp.stderr or ""))
    if not match:
        raise RuntimeError(f"Failed to parse sbatch output. stdout={cp.stdout!r} stderr={cp.stderr!r}")
    return int(match.group(1))


def scancel_job(jobid: int) -> None:
    run_cmd(["scancel", str(jobid)], check=True, capture=True)


def squeue_state(jobid: int) -> Optional[str]:
    cp = run_cmd(["squeue", "-j", str(jobid), "-h", "-o", "%T"], check=False, capture=True)
    if cp.returncode != 0:
        return None
    out = (cp.stdout or "").strip()
    if not out:
        return None
    return out.splitlines()[0].strip()


def sacct_info(jobid: int) -> Optional[dict]:
    fmt = "JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS"
    cp = run_cmd(["sacct", "-j", str(jobid), "--format", fmt, "-n", "-P"], check=False, capture=True)
    if cp.returncode != 0:
        return None
    lines = [ln.strip() for ln in (cp.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    for ln in lines:
        parts = ln.split("|")
        if len(parts) < 5:
            continue
        if parts[0].strip() == str(jobid):
            return {
                "JobIDRaw": parts[0].strip(),
                "State": parts[1].strip(),
                "ExitCode": parts[2].strip(),
                "ElapsedRaw": parts[3].strip(),
                "MaxRSS": parts[4].strip(),
            }
    return None


def resolve_request_job_id(job_id: str, job_dir: Path, req: dict, st: State) -> Optional[int]:
    raw_job_id = req.get("slurm_job_id")
    if raw_job_id is not None:
        try:
            return int(raw_job_id)
        except Exception:
            return None

    rec = st.active_jobs.get(job_id)
    if rec and "slurm_job_id" in rec:
        try:
            return int(rec["slurm_job_id"])
        except Exception:
            return None

    status = read_status(job_dir) or {}
    raw_job_id = status.get("slurm_job_id")
    if raw_job_id is None:
        return None
    try:
        return int(raw_job_id)
    except Exception:
        return None


def process_requests(cfg: Config, st: State) -> None:
    for req_path in list_request_paths(cfg):
        try:
            req = json.loads(req_path.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"[WORKER][WARN] invalid request file {req_path}: {e}")
            req_path.unlink(missing_ok=True)
            continue

        kind = str(req.get("type", "")).strip()
        job_id = str(req.get("job_id", "")).strip()
        job_dir = Path(str(req.get("job_dir", ""))).expanduser()
        request_git_sha = str(req.get("git_sha", "")).strip()
        requested_at = str(req.get("requested_at", "")).strip()

        if not job_id or not job_dir:
            log(f"[WORKER][WARN] dropping malformed request {req_path}")
            req_path.unlink(missing_ok=True)
            continue

        if kind == "submit_request":
            sbatch_script = Path(str(req.get("sbatch_script", ""))).expanduser()

            if job_id in st.active_jobs:
                req_path.unlink(missing_ok=True)
                continue

            try:
                jid = sbatch_submit(cfg, job_dir)
            except Exception as e:
                log(f"[WORKER][ERROR] submit failed for {job_id}: {e}")
                emit_result(
                    cfg,
                    "submit_failed",
                    job_id,
                    {
                        "job_id": job_id,
                        "job_dir": str(job_dir),
                        "sbatch_script": str(sbatch_script),
                        "git_sha": request_git_sha,
                        "requested_at": requested_at,
                        "failed_at": now_iso(),
                        "submit_error": str(e),
                        "state": "SUBMIT_FAILED",
                    },
                )
                req_path.unlink(missing_ok=True)
                continue

            submitted_at = now_iso()
            log_out = str((job_dir / f"slurm-{jid}.out").resolve())
            log_err = str((job_dir / f"slurm-{jid}.err").resolve())
            st.active_jobs[job_id] = {
                "job_dir": str(job_dir),
                "sbatch_script": str(sbatch_script),
                "slurm_job_id": jid,
                "submitted_at": submitted_at,
                "last_reported_state": "SUBMITTED",
            }
            save_state(cfg, st)
            emit_result(
                cfg,
                "submit_result",
                job_id,
                {
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "sbatch_script": str(sbatch_script),
                    "git_sha": request_git_sha,
                    "requested_at": requested_at,
                    "submitted_at": submitted_at,
                    "slurm_job_id": jid,
                    "state": "SUBMITTED",
                    "log_out": log_out,
                    "log_err": log_err,
                },
            )
            req_path.unlink(missing_ok=True)
            continue

        if kind != "cancel_request":
            log(f"[WORKER][WARN] dropping unknown request type {kind!r} from {req_path.name}")
            req_path.unlink(missing_ok=True)
            continue

        jid = resolve_request_job_id(job_id, job_dir, req, st)
        if jid is None:
            emit_result(
                cfg,
                "cancel_failed",
                job_id,
                {
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "git_sha": request_git_sha,
                    "requested_at": requested_at,
                    "failed_at": now_iso(),
                    "cancel_error": "missing slurm_job_id for cancel request",
                    "state": "CANCEL_FAILED",
                },
            )
            req_path.unlink(missing_ok=True)
            continue

        try:
            scancel_job(jid)
        except Exception as e:
            log(f"[WORKER][ERROR] cancel failed for {job_id} (job {jid}): {e}")
            emit_result(
                cfg,
                "cancel_failed",
                job_id,
                {
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "slurm_job_id": jid,
                    "git_sha": request_git_sha,
                    "requested_at": requested_at,
                    "failed_at": now_iso(),
                    "cancel_error": str(e),
                    "state": "CANCEL_FAILED",
                },
            )
            req_path.unlink(missing_ok=True)
            continue

        cancelled_at = now_iso()
        rec = st.active_jobs.get(job_id)
        if rec is not None:
            rec["cancel_requested_at"] = cancelled_at
            save_state(cfg, st)
        emit_result(
            cfg,
            "cancel_result",
            job_id,
            {
                "job_id": job_id,
                "job_dir": str(job_dir),
                "slurm_job_id": jid,
                "git_sha": request_git_sha,
                "requested_at": requested_at,
                "cancelled_at": cancelled_at,
                "state": "CANCEL_REQUESTED",
            },
        )
        req_path.unlink(missing_ok=True)


def refresh_active_jobs(cfg: Config, st: State) -> None:
    if not st.active_jobs:
        return

    changed = False
    done: List[str] = []

    for job_id, rec in list(st.active_jobs.items()):
        jid = int(rec["slurm_job_id"])
        sq = squeue_state(jid)
        if sq is not None:
            state_norm = normalize_state(sq)
            info = {"source": "squeue", "raw_state": sq}
        else:
            sa = sacct_info(jid)
            if sa is None:
                continue
            state_norm = normalize_state(sa.get("State", ""))
            info = {"source": "sacct", "raw_state": sa.get("State", ""), **sa}

        if state_norm and state_norm != rec.get("last_reported_state", ""):
            emit_result(
                cfg,
                "status_result",
                job_id,
                {
                    "job_id": job_id,
                    "job_dir": rec["job_dir"],
                    "slurm_job_id": jid,
                    "state": state_norm,
                    "last_update_at": now_iso(),
                    "slurm_info": info,
                },
            )
            rec["last_reported_state"] = state_norm
            changed = True

        if state_norm and is_terminal(state_norm):
            done.append(job_id)

    for job_id in done:
        st.active_jobs.pop(job_id, None)
        changed = True

    if changed:
        save_state(cfg, st)


def bootstrap_active_jobs(cfg: Config, st: State) -> None:
    for job_dir in list_ready_jobs(cfg):
        job_id = job_id_from_dir(job_dir)
        if job_id in st.active_jobs:
            continue
        status = read_status(job_dir)
        if not status or "slurm_job_id" not in status:
            continue
        try:
            jid = int(status["slurm_job_id"])
        except Exception:
            continue
        state_norm = normalize_state(str(status.get("state", "")))
        if state_norm and not is_terminal(state_norm):
            st.active_jobs[job_id] = {
                "job_dir": str(job_dir),
                "sbatch_script": str((job_dir / cfg.sbatch_name).resolve()),
                "slurm_job_id": jid,
                "submitted_at": str(status.get("submitted_at", "")),
                "last_reported_state": state_norm,
            }
    save_state(cfg, st)
    log(f"[WORKER][BOOTSTRAP] active_jobs={len(st.active_jobs)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Path to the git repo working tree.")
    p.add_argument("--jobs-dir", default="jobs", help="Jobs directory inside repo. Default: jobs")
    p.add_argument("--ready-name", default="READY", help="Ready marker file name. Default: READY")
    p.add_argument("--sbatch-name", default="run.sbatch", help="Sbatch script file name. Default: run.sbatch")
    p.add_argument("--request-interval", type=int, default=3, help="Seconds between scanning submit requests.")
    p.add_argument("--status-interval", type=int, default=15, help="Seconds between Slurm status refresh.")
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    repo_dir = Path(ns.repo).expanduser().resolve()
    broker_dir = repo_dir / ".broker"
    cfg = Config(
        repo_dir=repo_dir,
        jobs_dir=repo_dir / ns.jobs_dir,
        requests_dir=broker_dir / "requests",
        results_dir=broker_dir / "results",
        ready_name=ns.ready_name,
        sbatch_name=ns.sbatch_name,
        request_interval_s=max(1, int(ns.request_interval)),
        status_interval_s=max(3, int(ns.status_interval)),
        lock_file=repo_dir / ".gsync_worker.lock",
        state_file=repo_dir / ".gsync_worker_state.json",
        submit_as_user=detect_submit_as_user(repo_dir),
    )

    if not (cfg.repo_dir / ".git").exists():
        log(f"[WORKER][FATAL] Not a git repo: {cfg.repo_dir}")
        return 2

    cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
    cfg.requests_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    _lock_fd = acquire_lock(cfg.lock_file)
    submit_mode = cfg.submit_as_user or "<invoking-user>"
    log(f"[WORKER][START] pid={os.getpid()} repo={cfg.repo_dir} submit_as={submit_mode}")

    st = load_state(cfg)
    bootstrap_active_jobs(cfg, st)

    next_requests = 0.0
    next_status = 0.0

    while True:
        t = time.time()

        if t >= next_requests:
            try:
                process_requests(cfg, st)
            except Exception as e:
                log(f"[WORKER][WARN] request cycle error: {e}")
            next_requests = t + cfg.request_interval_s

        if t >= next_status:
            try:
                refresh_active_jobs(cfg, st)
            except Exception as e:
                log(f"[WORKER][WARN] status refresh error: {e}")
            next_status = t + cfg.status_interval_s

        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
