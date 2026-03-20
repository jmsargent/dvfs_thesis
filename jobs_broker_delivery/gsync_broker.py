#!/usr/bin/env python3
"""
gsync_broker.py — Normal-user sync broker.

This process is responsible for:
  - git pull / merge conflict handling
  - discovering new jobs under jobs/<job_id>/
  - writing submit requests into .broker/requests/
  - consuming worker results from .broker/results/
  - updating jobs/<job_id>/status.json and status.log
  - committing and pushing managed job files back to Git
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from broker_common import (
    BROKER_LOCAL_FILE_RE,
    CmdError,
    acquire_lock,
    append_status_log,
    atomic_write_json,
    clear_preexisting_managed_files,
    is_broker_generated_job_path,
    is_terminal,
    is_tracked_job_path,
    job_id_from_dir,
    log,
    normalize_state,
    now_iso,
    read_status,
    replace_status_log,
    run_cmd,
    status_log_path,
    write_status,
)


@dataclasses.dataclass
class Config:
    repo_dir: Path
    remote: str
    branch: str
    jobs_dir: Path
    broker_dir: Path
    requests_dir: Path
    results_dir: Path
    ready_name: str
    kill_name: str
    sbatch_name: str
    poll_interval_s: int
    result_interval_s: int
    push_interval_s: int
    git_recovery_failures: int
    lock_file: Path
    state_file: Path


@dataclasses.dataclass
class State:
    seen_jobs: Dict[str, dict]
    last_push_ts: float

    @staticmethod
    def empty() -> "State":
        return State(seen_jobs={}, last_push_ts=0.0)

    def to_json(self) -> dict:
        return {"seen_jobs": self.seen_jobs, "last_push_ts": self.last_push_ts}

    @staticmethod
    def from_json(data: dict) -> "State":
        return State(
            seen_jobs=dict(data.get("seen_jobs", {})),
            last_push_ts=float(data.get("last_push_ts", 0.0)),
        )


def load_state(cfg: Config) -> State:
    if cfg.state_file.exists():
        try:
            return State.from_json(json.loads(cfg.state_file.read_text(encoding="utf-8")))
        except Exception as e:
            log(f"[SYNC][WARN] failed to load state {cfg.state_file}: {e}; starting fresh")
    return State.empty()


def save_state(cfg: Config, st: State) -> None:
    atomic_write_json(cfg.state_file, st.to_json())


def git(
    cmd: List[str], cfg: Config, check: bool = True, capture: bool = True, env: Optional[dict] = None
) -> __import__("subprocess").CompletedProcess:
    return run_cmd(["git", *cmd], cwd=cfg.repo_dir, check=check, capture=capture, env=env)


def current_git_sha(cfg: Config) -> str:
    cp = git(["rev-parse", "HEAD"], cfg)
    return (cp.stdout or "").strip()


def git_porcelain(cfg: Config) -> List[str]:
    cp = git(["status", "--porcelain"], cfg, capture=True)
    return [ln for ln in (cp.stdout or "").splitlines() if ln]


def porcelain_path(ln: str) -> str:
    path = ln[3:] if len(ln) > 3 else ""
    if " -> " in path:
        return path.split(" -> ", 1)[1]
    return path


def repo_has_unexpected_changes(cfg: Config) -> Tuple[bool, List[str]]:
    unexpected: List[str] = []
    for ln in git_porcelain(cfg):
        path = porcelain_path(ln)
        if not is_tracked_job_path(path) and not BROKER_LOCAL_FILE_RE.match(path):
            unexpected.append(path)
    return (len(unexpected) > 0, unexpected)


def stage_managed_job_files(cfg: Config) -> None:
    paths: List[str] = []
    for ln in git_porcelain(cfg):
        path = porcelain_path(ln)
        if is_tracked_job_path(path):
            paths.append(path)
    if paths:
        git(["add", "--", *paths], cfg, capture=True)


def has_staged_changes(cfg: Config) -> bool:
    cp = git(["diff", "--cached", "--name-only"], cfg, capture=True)
    return bool((cp.stdout or "").strip())


def commit_managed_changes(cfg: Config, message: str) -> bool:
    stage_managed_job_files(cfg)
    if not has_staged_changes(cfg):
        return False
    git(["commit", "-m", message], cfg, capture=True)
    return True


def ensure_clean_before_pull(cfg: Config) -> None:
    bad, paths = repo_has_unexpected_changes(cfg)
    if bad:
        raise RuntimeError(
            "Repo has unexpected local changes (sync broker manages all files under jobs/* and broker local files).\n"
            "Run it in a dedicated clean clone.\n"
            f"Unexpected paths: {paths}"
        )
    commit_managed_changes(cfg, message=f"jobs: update status ({now_iso()})")


def git_unmerged_paths(cfg: Config) -> List[str]:
    cp = git(["diff", "--name-only", "--diff-filter=U"], cfg, check=False, capture=True)
    if cp.returncode != 0:
        return []
    return [ln.strip() for ln in (cp.stdout or "").splitlines() if ln.strip()]


def git_show_stage_blob(cfg: Config, stage: int, path: str) -> Optional[str]:
    cp = git(["show", f":{stage}:{path}"], cfg, check=False, capture=True)
    if cp.returncode != 0:
        return None
    return cp.stdout or ""


def make_conflict_copy_path(path: Path) -> Path:
    stamp = __import__("datetime").datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = path.with_name(f"{path.stem}.conflict-{stamp}{path.suffix}")
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.conflict-{stamp}-{idx}{path.suffix}")
        idx += 1
    return candidate


def resolve_merge_conflicts_by_copy(cfg: Config) -> bool:
    paths = git_unmerged_paths(cfg)
    if not paths:
        return False

    unmanaged = [path for path in paths if not is_broker_generated_job_path(path)]
    if unmanaged:
        log(f"[SYNC][WARN] cannot auto-resolve merge conflicts outside managed job files: {unmanaged}")
        return False

    resolved: List[Tuple[str, str, str]] = []
    for path in paths:
        local_content = git_show_stage_blob(cfg, 2, path)
        remote_content = git_show_stage_blob(cfg, 3, path)
        if remote_content is None or local_content is None:
            log(f"[SYNC][WARN] cannot auto-resolve non modify/modify conflict: {path}")
            return False
        resolved.append((path, remote_content, local_content))

    for path, remote_content, local_content in resolved:
        src_path = cfg.repo_dir / path
        copy_path = make_conflict_copy_path(src_path)
        copy_rel = str(copy_path.relative_to(cfg.repo_dir))
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_text(remote_content, encoding="utf-8")
        copy_path.write_text(local_content, encoding="utf-8")
        git(["add", "--", path, copy_rel], cfg, capture=True)
        log(f"[SYNC][WARN] merge conflict on {path}; kept remote version and copied local version to {copy_rel}")
    return True


def finish_merge(cfg: Config) -> None:
    env = dict(os.environ)
    env["GIT_EDITOR"] = "true"
    env["GIT_SEQUENCE_EDITOR"] = "true"
    env["GIT_MERGE_AUTOEDIT"] = "no"
    git(["commit", "--no-edit"], cfg, capture=True, env=env)


def git_pull_merge(cfg: Config) -> None:
    ensure_clean_before_pull(cfg)
    git(["checkout", cfg.branch], cfg, capture=True)
    env = dict(os.environ)
    env["GIT_MERGE_AUTOEDIT"] = "no"
    try:
        git(["pull", "--no-rebase", cfg.remote, cfg.branch], cfg, capture=True, env=env)
    except CmdError as e:
        if not resolve_merge_conflicts_by_copy(cfg):
            log(f"[SYNC][WARN] git pull failed; trying merge --abort.\n{e}")
            git(["merge", "--abort"], cfg, check=False, capture=True)
            raise
        try:
            finish_merge(cfg)
        except CmdError as merge_err:
            log(f"[SYNC][WARN] merge finalize failed; trying merge --abort.\n{merge_err}")
            git(["merge", "--abort"], cfg, check=False, capture=True)
            raise


def git_push(cfg: Config) -> None:
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            git_pull_merge(cfg)
        except Exception as e:
            raise RuntimeError(f"git pull failed before push:\n{e}") from e
        try:
            git(["push", cfg.remote, cfg.branch], cfg, capture=True)
            return
        except CmdError as e:
            if attempt >= max_attempts:
                raise RuntimeError(f"git push failed after retry:\n{e}") from e
            log(
                "[SYNC][WARN] git push failed after pull; "
                "remote likely advanced concurrently, retrying after another pull.\n"
                f"{e}"
            )


def git_hard_recover(cfg: Config) -> bool:
    log(
        "[SYNC][WARN] attempting hard git recovery; "
        "discarding local tracked and untracked changes in this clone."
    )
    git(["merge", "--abort"], cfg, check=False, capture=True)
    git(["rebase", "--abort"], cfg, check=False, capture=True)
    git(["reset", "--hard", "HEAD"], cfg, check=False, capture=True)
    git(["clean", "-fd"], cfg, check=False, capture=True)
    try:
        git(["fetch", cfg.remote, cfg.branch], cfg, capture=True)
        git(["checkout", cfg.branch], cfg, capture=True)
        git(["reset", "--hard", "FETCH_HEAD"], cfg, capture=True)
        git(["clean", "-fd"], cfg, capture=True)
        log(f"[SYNC][WARN] hard git recovery succeeded; repo reset to {cfg.remote}/{cfg.branch}")
        return True
    except Exception as e:
        log(f"[SYNC][WARN] hard git recovery failed:\n{e}")
        return False


def handle_git_sync_failure(cfg: Config, failure_count: int, context: str, err: Exception) -> Tuple[int, bool]:
    next_count = failure_count + 1
    log(
        f"[SYNC][WARN] {context}: {err}\n"
        f"[SYNC][WARN] consecutive git sync failures={next_count}/{cfg.git_recovery_failures}"
    )
    if next_count < cfg.git_recovery_failures:
        return next_count, False

    log(
        "[SYNC][WARN] git failure threshold reached; "
        "starting automatic hard recovery."
    )
    if git_hard_recover(cfg):
        return 0, True
    return next_count, False


def list_ready_jobs(cfg: Config) -> List[Path]:
    ready_files = glob.glob(str(cfg.jobs_dir / "*" / cfg.ready_name))
    jobs = [Path(p).parent for p in ready_files if Path(p).parent.is_dir()]
    jobs.sort()
    return jobs


def submit_request_path(cfg: Config, job_id: str) -> Path:
    return cfg.requests_dir / f"{job_id}.submit.json"


def cancel_request_path(cfg: Config, job_id: str) -> Path:
    return cfg.requests_dir / f"{job_id}.cancel.json"


def list_result_paths(cfg: Config) -> List[Path]:
    return sorted(p for p in cfg.results_dir.glob("*.json") if p.is_file())


def append_or_replace_status_log(job_dir: Path, line: str) -> None:
    if status_log_path(job_dir).exists():
        append_status_log(job_dir, line)
    else:
        replace_status_log(job_dir, line)


def queue_submit_request(cfg: Config, job_dir: Path, git_sha: str) -> None:
    job_id = job_id_from_dir(job_dir)
    req = {
        "type": "submit_request",
        "job_id": job_id,
        "job_dir": str(job_dir),
        "sbatch_script": str((job_dir / cfg.sbatch_name).resolve()),
        "git_sha": git_sha,
        "requested_at": now_iso(),
    }
    atomic_write_json(submit_request_path(cfg, job_id), req)


def queue_cancel_request(cfg: Config, job_dir: Path, slurm_job_id: int, git_sha: str) -> None:
    job_id = job_id_from_dir(job_dir)
    req = {
        "type": "cancel_request",
        "job_id": job_id,
        "job_dir": str(job_dir),
        "slurm_job_id": int(slurm_job_id),
        "git_sha": git_sha,
        "requested_at": now_iso(),
    }
    atomic_write_json(cancel_request_path(cfg, job_id), req)


def reconcile_pending_requests(cfg: Config, st: State, git_sha: str) -> None:
    for job_dir in list_ready_jobs(cfg):
        job_id = job_id_from_dir(job_dir)
        if job_id not in st.seen_jobs:
            continue
        if submit_request_path(cfg, job_id).exists():
            continue
        status = read_status(job_dir) or {}
        state_norm = normalize_state(str(status.get("state", "")))
        if state_norm != "QUEUED":
            continue
        if "slurm_job_id" in status:
            continue
        queue_submit_request(cfg, job_dir, git_sha)
        st.seen_jobs.setdefault(job_id, {})["request_queued_at"] = now_iso()
        save_state(cfg, st)
        log(f"[SYNC][INFO] re-queued pending submit request for {job_id}")


def reconcile_kill_requests(cfg: Config, st: State, git_sha: str) -> None:
    for job_dir in list_ready_jobs(cfg):
        job_id = job_id_from_dir(job_dir)
        if job_id not in st.seen_jobs:
            continue

        kill_path = job_dir / cfg.kill_name
        if not kill_path.exists():
            continue
        if cancel_request_path(cfg, job_id).exists():
            continue

        status = read_status(job_dir) or {}
        state_norm = normalize_state(str(status.get("state", "")))
        if is_terminal(state_norm):
            continue

        slurm_job_id_raw = status.get("slurm_job_id", st.seen_jobs.get(job_id, {}).get("slurm_job_id"))
        try:
            slurm_job_id = int(slurm_job_id_raw)
        except Exception:
            continue

        try:
            kill_mtime_ns = int(kill_path.stat().st_mtime_ns)
        except OSError:
            continue

        cancel_request_state = str(status.get("cancel_request_state", "")).strip().lower()
        cancel_kill_mtime_ns = status.get("cancel_kill_mtime_ns")

        if cancel_request_state == "accepted":
            continue
        if cancel_request_state == "failed":
            try:
                if int(cancel_kill_mtime_ns) == kill_mtime_ns:
                    continue
            except Exception:
                continue

        first_request_for_marker = False
        try:
            first_request_for_marker = int(cancel_kill_mtime_ns) != kill_mtime_ns
        except Exception:
            first_request_for_marker = True

        new = dict(status)
        if first_request_for_marker or not str(new.get("cancel_requested_at", "")).strip():
            new["cancel_requested_at"] = now_iso()
        new["cancel_request_state"] = "queued"
        new["cancel_kill_mtime_ns"] = kill_mtime_ns
        new["last_update_at"] = now_iso()
        write_status(job_dir, new)
        if first_request_for_marker or cancel_request_state not in ("queued", "accepted"):
            append_or_replace_status_log(job_dir, f"CANCEL_REQUESTED slurm_job_id={slurm_job_id} git_sha={git_sha}")

        queue_cancel_request(cfg, job_dir, slurm_job_id, git_sha)
        st.seen_jobs.setdefault(job_id, {})["cancel_requested_at"] = str(new["cancel_requested_at"])
        save_state(cfg, st)
        log(f"[SYNC][INFO] queued cancel request for {job_id} (job {slurm_job_id})")


def discover_new_jobs(cfg: Config, st: State) -> None:
    git_pull_merge(cfg)
    git_sha = current_git_sha(cfg)

    for job_dir in list_ready_jobs(cfg):
        job_id = job_id_from_dir(job_dir)
        if job_id in st.seen_jobs:
            continue

        sbatch_script = job_dir / cfg.sbatch_name
        if not sbatch_script.exists():
            log(f"[SYNC][WARN] new job {job_id} has READY but missing {cfg.sbatch_name}; skipping")
            continue

        removed = clear_preexisting_managed_files(job_dir)
        if removed:
            log(f"[SYNC][WARN] cleared copied managed files for new job {job_id}: {removed}")

        status = {
            "id": job_id,
            "git_sha": git_sha,
            "job_dir": str(job_dir),
            "sbatch_script": str(sbatch_script.resolve()),
            "queued_at": now_iso(),
            "last_update_at": now_iso(),
            "state": "QUEUED",
        }
        write_status(job_dir, status)
        replace_status_log(job_dir, f"QUEUED git_sha={git_sha}")
        queue_submit_request(cfg, job_dir, git_sha)

        st.seen_jobs[job_id] = {
            "first_seen_at": now_iso(),
            "request_queued_at": now_iso(),
        }
        save_state(cfg, st)

    reconcile_pending_requests(cfg, st, git_sha)
    reconcile_kill_requests(cfg, st, git_sha)


def apply_submit_result(cfg: Config, st: State, result: dict) -> bool:
    job_id = str(result["job_id"])
    job_dir = Path(str(result["job_dir"]))
    cur = read_status(job_dir) or {}
    state_norm = normalize_state(str(cur.get("state", "")))
    jid = int(result["slurm_job_id"])

    new = dict(cur)
    new["id"] = job_id
    new["git_sha"] = str(result.get("git_sha", "")) or str(cur.get("git_sha", ""))
    new["submitted_at"] = str(result["submitted_at"])
    new["slurm_job_id"] = jid
    if state_norm in ("", "QUEUED", "SUBMIT_FAILED", "SUBMITTED"):
        new["state"] = "SUBMITTED"
    new["log_out"] = str(result["log_out"])
    new["log_err"] = str(result["log_err"])
    new["job_dir"] = str(job_dir)
    new["sbatch_script"] = str(result["sbatch_script"])
    new["last_update_at"] = now_iso()

    updated = write_status(job_dir, new)
    if int(cur.get("slurm_job_id", -1)) != jid:
        append_or_replace_status_log(
            job_dir,
            f"SUBMITTED slurm_job_id={jid} git_sha={new.get('git_sha', '')}",
        )
    st.seen_jobs.setdefault(job_id, {})["slurm_job_id"] = jid
    save_state(cfg, st)
    return updated


def apply_submit_failed(cfg: Config, st: State, result: dict) -> bool:
    job_id = str(result["job_id"])
    job_dir = Path(str(result["job_dir"]))
    cur = read_status(job_dir) or {}
    err = str(result.get("submit_error", "submit failed"))

    new = dict(cur)
    new["id"] = job_id
    new["git_sha"] = str(result.get("git_sha", "")) or str(cur.get("git_sha", ""))
    new["job_dir"] = str(job_dir)
    new["sbatch_script"] = str(result["sbatch_script"])
    new["last_update_at"] = now_iso()
    new["state"] = "SUBMIT_FAILED"
    new["submit_failed"] = True
    new["submit_error"] = err

    updated = write_status(job_dir, new)
    append_or_replace_status_log(job_dir, f"SUBMIT_FAILED: {err}")
    st.seen_jobs.setdefault(job_id, {})["submit_failed"] = True
    save_state(cfg, st)
    return updated


def apply_cancel_result(cfg: Config, st: State, result: dict) -> bool:
    job_id = str(result["job_id"])
    job_dir = Path(str(result["job_dir"]))
    cur = read_status(job_dir) or {}
    jid = int(result["slurm_job_id"])

    new = dict(cur)
    new["id"] = job_id
    new["slurm_job_id"] = jid
    new["cancel_requested_at"] = str(result.get("requested_at", "")) or str(cur.get("cancel_requested_at", ""))
    new["cancel_acknowledged_at"] = str(result.get("cancelled_at", "")) or now_iso()
    new["cancel_request_state"] = "accepted"
    new["last_update_at"] = now_iso()
    if not is_terminal(normalize_state(str(cur.get("state", "")))):
        new["state"] = "CANCEL_REQUESTED"

    updated = write_status(job_dir, new)
    append_or_replace_status_log(job_dir, f"CANCEL_SENT slurm_job_id={jid}")
    st.seen_jobs.setdefault(job_id, {})["cancel_acknowledged_at"] = str(new["cancel_acknowledged_at"])
    save_state(cfg, st)
    return updated


def apply_cancel_failed(cfg: Config, st: State, result: dict) -> bool:
    job_id = str(result["job_id"])
    job_dir = Path(str(result["job_dir"]))
    cur = read_status(job_dir) or {}
    jid = result.get("slurm_job_id")
    err = str(result.get("cancel_error", "cancel failed"))

    new = dict(cur)
    new["id"] = job_id
    if jid is not None:
        try:
            new["slurm_job_id"] = int(jid)
        except Exception:
            pass
    new["cancel_requested_at"] = str(result.get("requested_at", "")) or str(cur.get("cancel_requested_at", ""))
    new["cancel_failed_at"] = str(result.get("failed_at", "")) or now_iso()
    new["cancel_request_state"] = "failed"
    new["cancel_error"] = err
    new["last_update_at"] = now_iso()

    updated = write_status(job_dir, new)
    append_or_replace_status_log(job_dir, f"CANCEL_FAILED: {err}")
    st.seen_jobs.setdefault(job_id, {})["cancel_failed_at"] = str(new["cancel_failed_at"])
    save_state(cfg, st)
    return updated


def apply_status_result(result: dict) -> bool:
    job_dir = Path(str(result["job_dir"]))
    job_id = str(result["job_id"])
    jid = int(result["slurm_job_id"])
    state_norm = normalize_state(str(result.get("state", "")))
    cur = read_status(job_dir) or {}
    prev = normalize_state(str(cur.get("state", "")))

    new = dict(cur)
    new.setdefault("id", job_id)
    new["slurm_job_id"] = jid
    new["state"] = state_norm or new.get("state", "")
    new["last_update_at"] = str(result.get("last_update_at", now_iso()))
    new["slurm_info"] = dict(result.get("slurm_info", {}))

    updated = write_status(job_dir, new)
    if state_norm and state_norm != prev:
        append_or_replace_status_log(job_dir, f"STATE {prev or '?'} -> {state_norm} (job {jid})")
    return updated


def process_results(cfg: Config, st: State) -> bool:
    any_updated = False
    for path in list_result_paths(cfg):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"[SYNC][WARN] invalid result file {path}: {e}")
            path.unlink(missing_ok=True)
            continue

        kind = str(result.get("type", "")).strip()
        try:
            if kind == "submit_result":
                updated = apply_submit_result(cfg, st, result)
            elif kind == "submit_failed":
                updated = apply_submit_failed(cfg, st, result)
            elif kind == "cancel_result":
                updated = apply_cancel_result(cfg, st, result)
            elif kind == "cancel_failed":
                updated = apply_cancel_failed(cfg, st, result)
            elif kind == "status_result":
                updated = apply_status_result(result)
            else:
                log(f"[SYNC][WARN] ignoring unknown result type {kind!r} in {path.name}")
                updated = False
        except Exception as e:
            log(f"[SYNC][WARN] failed to apply result {path.name}: {e}")
            continue

        any_updated = any_updated or updated
        path.unlink(missing_ok=True)

    if any_updated:
        save_state(cfg, st)
    return any_updated


def bootstrap_seen_jobs(cfg: Config, st: State) -> None:
    git_pull_merge(cfg)
    for job_dir in list_ready_jobs(cfg):
        job_id = job_id_from_dir(job_dir)
        st.seen_jobs.setdefault(job_id, {"bootstrapped_at": now_iso()})
    save_state(cfg, st)
    log(f"[SYNC][BOOTSTRAP] seen_jobs={len(st.seen_jobs)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Path to the git repo working tree.")
    p.add_argument("--remote", default="origin", help="Git remote name. Default: origin")
    p.add_argument("--branch", default="main", help="Branch to track. Default: main")
    p.add_argument("--jobs-dir", default="jobs", help="Jobs directory inside repo. Default: jobs")
    p.add_argument("--ready-name", default="READY", help="Ready marker file name. Default: READY")
    p.add_argument("--kill-name", default="KILL", help="Cancel marker file name. Default: KILL")
    p.add_argument("--sbatch-name", default="run.sbatch", help="Sbatch script file name. Default: run.sbatch")
    p.add_argument("--poll-interval", type=int, default=10, help="Seconds between git pull + discovery. Default: 10")
    p.add_argument(
        "--result-interval",
        "--status-interval",
        dest="result_interval",
        type=int,
        default=5,
        help="Seconds between consuming worker result files. Default: 5",
    )
    p.add_argument("--push-interval", type=int, default=30, help="Seconds between pushing status updates.")
    p.add_argument(
        "--git-recovery-failures",
        type=int,
        default=5,
        help="Consecutive git sync failures before hard-resetting the local clone to remote. Default: 5",
    )
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    repo_dir = Path(ns.repo).expanduser().resolve()
    broker_dir = repo_dir / ".broker"
    cfg = Config(
        repo_dir=repo_dir,
        remote=ns.remote,
        branch=ns.branch,
        jobs_dir=repo_dir / ns.jobs_dir,
        broker_dir=broker_dir,
        requests_dir=broker_dir / "requests",
        results_dir=broker_dir / "results",
        ready_name=ns.ready_name,
        kill_name=ns.kill_name,
        sbatch_name=ns.sbatch_name,
        poll_interval_s=max(3, int(ns.poll_interval)),
        result_interval_s=max(1, int(ns.result_interval)),
        push_interval_s=max(10, int(ns.push_interval)),
        git_recovery_failures=max(1, int(ns.git_recovery_failures)),
        lock_file=repo_dir / ".gsync_broker.lock",
        state_file=repo_dir / ".gsync_broker_state.json",
    )

    if not (cfg.repo_dir / ".git").exists():
        log(f"[SYNC][FATAL] Not a git repo: {cfg.repo_dir}")
        return 2

    cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
    cfg.requests_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    _lock_fd = acquire_lock(cfg.lock_file)
    log(f"[SYNC][START] pid={os.getpid()} repo={cfg.repo_dir} branch={cfg.branch}")

    st = load_state(cfg)
    bootstrapped = bool(st.seen_jobs)
    git_failure_count = 0

    next_poll = 0.0
    next_results = 0.0
    next_push = 0.0

    while True:
        t = time.time()

        if not bootstrapped:
            try:
                bootstrap_seen_jobs(cfg, st)
                git_failure_count = 0
                bootstrapped = True
                next_poll = t
                next_results = t
                next_push = t
            except Exception as e:
                git_failure_count, recovered = handle_git_sync_failure(cfg, git_failure_count, "bootstrap error", e)
                if recovered:
                    next_poll = t
                    next_results = t
                    next_push = t
                    time.sleep(1.0)
                    continue
                next_poll = t + cfg.poll_interval_s
                next_results = t + cfg.result_interval_s
                next_push = t + cfg.push_interval_s
                time.sleep(1.0)
                continue

        if t >= next_results:
            try:
                process_results(cfg, st)
            except Exception as e:
                log(f"[SYNC][WARN] result processing error: {e}")
            next_results = t + cfg.result_interval_s

        if t >= next_poll:
            try:
                discover_new_jobs(cfg, st)
                git_failure_count = 0
            except Exception as e:
                git_failure_count, recovered = handle_git_sync_failure(
                    cfg, git_failure_count, "discover cycle error", e
                )
                if recovered:
                    next_poll = t
                    next_results = t
                    next_push = t
            next_poll = t + cfg.poll_interval_s

        if t >= next_push:
            try:
                git_push(cfg)
                git_failure_count = 0
                st.last_push_ts = time.time()
                save_state(cfg, st)
            except Exception as e:
                git_failure_count, recovered = handle_git_sync_failure(cfg, git_failure_count, "push error", e)
                if recovered:
                    next_poll = t
                    next_results = t
                    next_push = t
            next_push = t + cfg.push_interval_s

        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
