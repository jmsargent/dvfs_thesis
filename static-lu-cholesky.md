Here's the corrected specification:

---

# Specification: Static Multi-GPU Single-Node Scheduling for Mustard

## Background

Mustard (`lu_mustard.cu`, `cholesky_mustard.cu`) is a CUDA Graph-based linear algebra benchmark. It already supports multi-GPU execution via `--subgraph` mode, which uses **dynamic** device-side scheduling: all PEs share a work queue (via NVSHMEM atomics), and tasks are dispatched at runtime based on GPU occupancy.

The goal is to add a **new** `--static-multigpu` flag that implements static multi-GPU scheduling on a single node, where task-to-GPU assignment is fixed at setup time, and inter-GPU signaling uses **NVSHMEM put/get** (no atomics). All existing flags and code paths (`--tiled`, `--subgraph`) remain completely untouched.

---

## What Already Exists (Do Not Touch)

- `TiledGraphCreator` — builds per-task sub-graphs and computes `subgraphDependencies`
- `nvshmem_malloc` / `nvshmem_ptr` — shared memory space across PEs
- `cudaMemcpy2DAsync` tile transfer logic between PEs
- `cudaGraphInstantiate` with `cudaGraphInstantiateFlagDeviceLaunch`
- `kernel_occupancy_update`, `kernel_scheduler`, `kernel_populate_queue`, `kernel_dep_update`, `kernel_dep_wait`
- `BrokerWorkDistributor` queue
- All existing `--tiled` and `--subgraph` code paths

---

## What Needs to Be Built

### 1. New CLI Flag

In `include/cli.h`, add parsing for `--static-multigpu`. In `MustardConfig` add:

```cpp
bool staticMultiGPU = false;
```

Parse it alongside the existing flags in `parseCommonArgs`.

### 2. New Dispatch in `LU()` and `Cholesky()`

Add a new branch **without removing any existing branch**:

```cpp
void LU(bool tiled, bool verify, bool subgraph, bool staticMultiGPU, bool dot)
{
    if (staticMultiGPU)
        tiledLUStatic(verify, dot);       // new path, all PEs enter
    else if (tiled && myPE == 0)
        tiledLU(verify, false, dot);      // existing
    else if (subgraph)
        tiledLU(verify, true, dot);       // existing
    else if (myPE == 0)
        trivialLU(verify);                // existing
}
```

### 3. New Function `tiledLUStatic()` (and `tiledCholeskyStatic()`)

This is a new function, separate from `tiledLU`. It reuses the graph construction infrastructure but replaces the scheduler. Structure:

#### 3a. Graph Construction (reuse existing)

The graph construction loop (GETRF/TRSM/GEMM capture with `TiledGraphCreator`) is **identical** to the `--subgraph` path in `tiledLU`, including the `cudaMemcpy2DAsync` tile transfers and `kernel_occupancy_update` calls. Copy it verbatim — do not share code with `tiledLU` to keep paths independent and avoid breaking existing functionality.

#### 3b. Static Task Assignment

After graph construction, on the host, assign each task index to a PE using round-robin:

```cpp
int nPEs = nvshmem_n_pes();
std::vector<std::vector<int>> pe_tasks(nPEs);
for (int i = 0; i < totalNodes; i++)
    pe_tasks[i % nPEs].push_back(i);
```

Then for each PE's task list, produce a **topologically sorted** execution order respecting `subgraphDependencies`. A simple approach: iterate through `pe_tasks[myPE]` and defer any task whose dependencies are not yet completed, repeating until all tasks are ordered.

#### 3c. Completion Flag Array

Allocate via NVSHMEM so all PEs can write to each other's flags:

```cpp
int *d_completion_flags = (int *) nvshmem_malloc(sizeof(int) * totalNodes);
cudaMemset(d_completion_flags, 0, sizeof(int) * totalNodes);
```

#### 3d. New Device Kernels (add to `mustard.h`, do not remove existing kernels)

**Signal completion** — called after a task finishes, writes to all PEs that have a dependent task:

```cuda
__global__ void kernel_signal_static(int task_id, 
                                      int* d_completion_flags,
                                      int* d_notify_pes, 
                                      int n_notify_pes)
{
    int one = 1;
    nvshmem_fence();
    for (int i = 0; i < n_notify_pes; i++) {
        nvshmem_int_put(&d_completion_flags[task_id], &one, 1, d_notify_pes[i]);
    }
}
```

**Wait on dependencies** — spin-polls local copy of `d_completion_flags`:

```cuda
__global__ void kernel_wait_static(int* d_deps, int n_deps, int* d_completion_flags)
{
    for (int i = 0; i < n_deps; i++) {
        while (d_completion_flags[d_deps[i]] == 0) { }
    }
}
```

#### 3e. Inject Signal and Wait Kernels into Sub-graphs

After `TiledGraphCreator` finishes building all sub-graphs (same point where `insertDependencyKernel` is called in the `--subgraph` path), for each task `i` owned by this PE:

- Prepend `kernel_wait_static` to `subgraphs[i]` — waiting on all dependencies of task `i`
- Append `kernel_signal_static` to `subgraphs[i]` — notifying all PEs that have tasks depending on task `i`

For the notify list: iterate `subgraphDependencies` to find which tasks depend on task `i`, then map those tasks to their assigned PEs via the round-robin assignment.

#### 3f. Per-PE Execution Loop

Each PE iterates its topologically sorted task list and launches sub-graphs sequentially on its stream:

```cpp
for (int task_i : my_tasks_sorted) {
    cudaGraphLaunch(h_subgraphsExec[task_i], s);
}
cudaStreamSynchronize(s);
```

No queue, no occupancy check, no shared counter.

#### 3g. Reset Between Runs

Before each timing run, reset flags to 0 on all PEs:

```cpp
nvshmem_barrier_all();
if (myPE == 0) {
    std::vector<int> zeros(totalNodes, 0);
    for (int pe = 0; pe < nPEs; pe++)
        nvshmem_int_put(d_completion_flags, zeros.data(), totalNodes, pe);
    nvshmem_quiet();
}
nvshmem_barrier_all();
```

---

## Key Constraints

- `nvshmem_fence()` must be called before `nvshmem_int_put` in `kernel_signal_static` to ensure the sub-graph's compute results are visible before the flag is written.
- `nvshmem_quiet()` on the host reset path ensures all puts have landed before execution begins.
- Spin-poll in `kernel_wait_static` reads **local** memory — correct because `nvshmem_int_put` from the completing PE wrote into this PE's local `d_completion_flags`.
- The `MAX_TILE` / `insertSubgraph` path for large LU tiles is already inside the graph construction loop that is copied verbatim — leave it unchanged.
- `N` must be divisible by `T` (existing constraint, unchanged).
- All existing `nvshmem_malloc` allocations for `d_matrices` and tile transfers are reused unchanged.

---

## Files to Modify

- `mustard/lu_mustard.cu` — add `tiledLUStatic()`, update `LU()` dispatch
- `mustard/cholesky_mustard.cu` — add `tiledCholeskyStatic()`, update `Cholesky()` dispatch
- `mustard/mustard.h` — add `kernel_signal_static` and `kernel_wait_static`, keep all existing kernels
- `include/cli.h` — add `--static-multigpu` flag and `staticMultiGPU` field to `MustardConfig`

---