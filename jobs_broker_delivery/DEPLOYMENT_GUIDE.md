# gsync Deployment & Setup Guide

This guide provides step-by-step instructions for setting up `gsync` to run Slurm jobs from a supervisor's account using your own credentials for Git synchronization.

## 1. Architecture Overview

`gsync` consists of two decoupled processes that communicate via files in the `.broker/` directory of a target Git repository:

1.  **Broker (`gsync_broker.py`):** Runs as a **normal user** (e.g., the student). It handles all Git operations (`pull`, `commit`, `push`) and watches for new job directories.
2.  **Worker (`gsync_worker.py`):** Usually runs as **root** (via `sudo`). It executes Slurm commands (`sbatch`, `squeue`, etc.). 

**The Key Trick:** When the Worker runs as `root`, it detects the owner of the local repository clone and automatically appends `--uid=<owner>` to `sbatch` commands. This allows it to submit jobs as the supervisor if the supervisor owns the repository clone.

---

## 2. Target Repository Requirements

The "Target Repository" is the Git repo where you will manage your jobs. It must satisfy these conditions:

### 2.1 Directory Structure
The broker expects a `jobs/` directory at the root:
```text
<repo-root>/
  jobs/
    <job_id>/
      run.sbatch     # The Slurm script to execute
      READY          # Marker file created LAST to trigger submission
```

### 2.2 Git Configuration
The broker needs to perform automated commits and pushes. 
- Ensure a `.gitignore` exists at the root with these entries:
  ```text
  .broker/
  .gsync_broker.lock
  .gsync_broker_state.json
  .gsync_worker.lock
  .gsync_worker_state.json
  ```
- The local clone used by the broker should have a git user configured:
  ```bash
  git config user.name "gsync-broker"
  git config user.email "gsync-broker@local"
  ```

### 2.3 Job Protocol
- **Unique Job IDs:** Each job must have a unique directory name under `jobs/`.
- **READY Marker:** The `READY` file must be created *after* `run.sbatch` is fully written and closed. The broker only looks for directories containing a `READY` file.
- **KILL Marker (Optional):** Creating a `KILL` file inside a job directory triggers an `scancel` for that job.

---

## 3. Step-by-Step Setup (Supervisor Account)

To run jobs as your supervisor, follow these steps on a machine that has access to the Slurm cluster.

### Step 1: Prepare the Local Clone
The local clone **must be owned by the supervisor** for the `--uid` trick to work.

1.  Ask your supervisor to clone the target repository to a specific location on the cluster's login node (or a shared filesystem):
    ```bash
    # Run as supervisor
    git clone <your-repo-url> /path/to/supervisor-owned-gsync-repo
    ```
2.  Ensure you (the student) have **read/write access** to this directory so you can run the broker.

### Step 2: Configure SSH Keys
The Broker (running as you) needs to push/pull from Git.
- Ensure your SSH keys are set up on the machine so `git pull` and `git push` work without a password.

### Step 3: Start the Broker (As Student)
The broker handles Git sync. Run it as your normal user:

```bash
cd /path/to/gsync-code
./gsync.sh start /path/to/supervisor-owned-gsync-repo
```

Check the status:
```bash
./gsync.sh status /path/to/supervisor-owned-gsync-repo
```

### Step 4: Start the Worker (As Root)
The worker needs root privileges to use `sbatch --uid`.

```bash
cd /path/to/gsync-code
sudo ./gsync_worker.sh start /path/to/supervisor-owned-gsync-repo
```

Check the status:
```bash
sudo ./gsync_worker.sh status /path/to/supervisor-owned-gsync-repo
```

---

## 4. Operational Notes

### Logs
Logs are stored inside the target repository's `.broker/` directory:
- `broker.log`: Git sync and job discovery events.
- `slurm_worker.log`: Slurm submission, cancellation, and status updates.

### Recovery
If Git synchronization fails repeatedly (e.g., due to complex merge conflicts), the broker will eventually attempt a "hard recovery" by running `git reset --hard` and `git clean -fd`. **This is why the local clone should be dedicated to gsync and not used for manual editing.**

### Slurm Permissions
The user running the worker (root) must have permission to submit jobs on behalf of other users in Slurm. This is standard in most Slurm configurations for root.
