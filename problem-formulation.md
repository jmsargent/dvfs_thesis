# Problem Formulation: Dynamic Subgraph Scheduler Crash

## Overview

The `--subgraph` mode of `cholesky_mustard` has **never successfully run**. Every attempt since job 052 crashes with:

```
CUDA error at mustard/mustard/cholesky_mustard.cu:440 code=700 "cudaDeviceSynchronize()"
```

After device 0 crashes and exits, devices 1/2/3 continue into their scheduler loops and hang waiting for PE 0's queue — causing the Slurm job to hit its wall-clock timeout.

---

## What the Subgraph Mode Is Supposed to Do

Instead of building one monolithic CUDA graph that bakes in all tile dependencies statically, the subgraph mode:

1. Captures each Cholesky tile operation (POTRF, TRSM, SYRK/GEMM) as its own small subgraph.
2. Builds a dependency graph over those subgraphs (`tiledCholeskyGraphCreator->subgraphDependencies`).
3. At runtime, a persistent scheduler kernel (`kernel_scheduler`) dequeues "ready" subgraphs from a shared NVSHMEM-backed queue (`BrokerWorkDistributor`) and fires them off with `cudaGraphLaunch(..., cudaStreamGraphFireAndForget)`.
4. When a subgraph finishes, it executes a `kernel_dep_update` graph node that decrements the dependency counter for each downstream node and enqueues any node whose counter reaches zero.
5. PE 0 owns the queue. All PEs have their own scheduler and subgraphs but coordinate through PE 0's NVSHMEM memory.

---

## The Crash Location

`cholesky_mustard.cu:440`:

```cpp
if (myPE == 0)
    mustard::kernel_populate_queue<<<108, 1024>>>(queue, d_dependencies, totalNodes);
checkCudaErrors(cudaDeviceSynchronize());   // <-- line 440, error caught here
```

`kernel_populate_queue` scans `d_dependencies` for nodes with zero incoming dependencies and enqueues them into `BrokerWorkDistributor` to seed the scheduler. With 20 nodes and only node 0 having zero deps, only a single enqueue ever actually fires.

---

## Root Cause: Confirmed by compute-sanitizer (job 057)

Running job 057 with `compute-sanitizer --tool memcheck` produced:

```
========= Invalid __global__ atomic of size 8 bytes
=========     at nvshmemi_transfer_amo_fetch<unsigned int>(...)+0x11b20
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x0 is out of bounds
=========     Device Frame: mustard::kernel_populate_queue(...)
```

**The faulting address is `0x0` — a null pointer.** The null is not in `count`/`tail`/`tickets` themselves (those are validly allocated NVSHMEM memory). The null is inside NVSHMEM's own internal transfer-path state, which `kernel_populate_queue` tries to use but finds uninitialized.

---

## Why NVSHMEM Atomics Fail in a Host-Launched Kernel

### How NVSHMEM device-side atomics work

NVSHMEM device-side operations like `nvshmem_int_atomic_fetch(ptr, pe)` are not simple wrappers around CUDA atomics. Internally they go through one of two paths:

- **P2P (direct) path**: uses NVLink/peer-access to directly issue an atomic on the target GPU's memory. Only available when the GPU hardware supports native cross-device atomics and the memory is peer-mapped.
- **Transfer (proxy) path** (`nvshmemi_transfer_amo_fetch`): sends the atomic request through a NVSHMEM software proxy. The device-side code writes the request into a "proxy channel" — a shared memory region between the GPU and a CPU-side proxy thread. The CPU proxy thread then performs the operation and writes the result back.

The proxy channel address is a device-accessible pointer that NVSHMEM stores in a **per-kernel device-side state block**. This state block is set up by NVSHMEM's kernel launch machinery.

### Why the proxy channel is null

The critical distinction is **how the kernel is launched**:

| Launch method | NVSHMEM per-kernel state | Transfer path works? |
|---|---|---|
| `nvshmemx_collective_launch(...)` | Initialized by NVSHMEM | Yes |
| Device-launched graph (`cudaGraphInstantiateFlagDeviceLaunch`) | Inherited from parent kernel context | Yes |
| Host `<<<>>>` | **Not set up** | **No** — proxy channel pointer is null |

When NVSHMEM routes `nvshmem_int_atomic_fetch(count, 0)` through `nvshmemi_transfer_amo_fetch`, it dereferences the proxy channel pointer to post the request. Since that pointer is null for a host-launched kernel, the GPU faults at address `0x0`.

### Why this matters even for pe=0 (local PE)

You might expect that pe=0 on PE 0 would be a purely local operation and bypass the proxy entirely. In practice, NVSHMEM does not always take the "obviously local" shortcut. Depending on how the NVSHMEM library was compiled and which transport is active (`NVSHMEM_REMOTE_TRANSPORT=none` disables InfiniBand but still leaves the proxy path as the fallback for atomics), NVSHMEM routes even local-PE atomics through `nvshmemi_transfer_amo_fetch`. The result is the same null-pointer fault regardless of whether `pe == myPE`.

### Why kernel_dep_update and kernel_scheduler work

Both `kernel_dep_update` and `kernel_scheduler` also use NVSHMEM device-side atomics — and they work. The difference is how they are launched:

- `kernel_dep_update` is added as a node inside a CUDA subgraph instantiated with `cudaGraphInstantiateFlagDeviceLaunch`. It is launched from device code within a kernel that already has NVSHMEM context. The NVSHMEM per-kernel state is inherited down the launch chain.
- `kernel_scheduler` is captured into a CUDA graph via `cudaStreamBeginCapture`/`cudaStreamEndCapture`, instantiated with `cudaGraphInstantiateFlagDeviceLaunch`, and launched from device code. Same story.
- `kernel_populate_queue` is the **only** kernel launched directly from the host via `<<<>>>`. It never gets the NVSHMEM per-kernel state.

---

## Jobs and Attempts

| Job | git SHA | Key change | Outcome |
|-----|---------|-----------|---------|
| 047–049 | — | Printing subgraph structure | Printed OK, no actual compute |
| 050 | — | First `--subgraph` run | FATAL: wrong container path |
| 051 | — | Corrected paths, but used `--tiled` (not `--subgraph`) | SUCCESS — tiled mode works |
| 052 | `0ba39ae` | First real `--subgraph` run, N=24000, t=8 | FAIL: code 700 at line 440 |
| 053–054 | — | Retries | FAIL: code 700 at line 440 |
| 055 | `562b730` | Added `cudaMemset` zero-init to `BrokerWorkDistributor` (commit `68635b9`), rebuilt binary | FAIL: code 700 at line 440 |
| 056 | — | Added `compute-sanitizer --tool memcheck` to sbatch | — |
| **057** | — | compute-sanitizer run | **Confirmed: null proxy channel in `nvshmemi_transfer_amo_fetch`** |

The `cudaMemset` initialization (job 055) fixed a separate latent bug — uninitialized `tickets[]` would have caused `waitForTicket` to spin forever on any successful queue operation. The code 700 crash predates that change and is unrelated to initialization.

---

## What Works vs. What Doesn't

- `--tiled` mode (static single CUDA graph): **works** — no NVSHMEM device-side ops in host-launched kernels.
- `--subgraph` mode: **never worked** — `kernel_populate_queue` is host-launched and uses NVSHMEM atomics.

---

## Proposed Fix

### Option A: Replace NVSHMEM atomics with CUDA atomics in `kernel_populate_queue` (recommended)

`kernel_populate_queue` always runs only on PE 0 (`if (myPE == 0)`) and always passes `pe=0`. Every operation is a local read/write to PE 0's own memory. The NVSHMEM semantics add nothing here — there is no cross-GPU communication in this kernel.

The fix is to add an `enqueue_local` method to `BrokerWorkDistributor` that implements the same ring-buffer protocol using plain CUDA atomics and volatile memory accesses, and have `kernel_populate_queue` call it instead of `enqueue`.

NVSHMEM-allocated memory is regular CUDA device memory. CUDA atomics work on it. The data written by `enqueue_local` is then visible to `kernel_dep_update` and `kernel_scheduler` via their NVSHMEM ops, because those kernels run in the NVSHMEM device-launched graph context and can correctly access PE 0's symmetric heap.

Operation mapping (all valid for local PE only):

```cpp
nvshmem_int_atomic_fetch(count, 0)        →  atomicAdd((int*)count, 0)
nvshmem_int_atomic_fetch_add(count, ±1, 0) →  atomicAdd((int*)count, ±1)
nvshmem_uint_atomic_fetch_inc(tail, 0)    →  atomicAdd(tail, 1u)
nvshmem_uint_g(&tickets[P], 0)            →  *(volatile Ticket*)&tickets[P]
nvshmem_uint_p(&ring_buffer[P], v, 0)     →  ring_buffer[P] = v
nvshmem_uint_p(&tickets[P], v, 0)         →  *(volatile Ticket*)&tickets[P] = v
```

`__threadfence()` replaces `nvshmem_fence()` for the memory ordering between `ring_buffer[P] = data` and the ticket write.

### Option B: Use `nvshmemx_collective_launch` instead of `<<<>>>`

`nvshmemx_collective_launch` is NVSHMEM's host-side collective launch API that properly sets up the per-kernel NVSHMEM state. All PEs must call it. `kernel_populate_queue` only does work on PE 0 (`if (myPE == 0)`) so other PEs would return immediately, but all PEs must participate in the launch. This is the more "correct" NVSHMEM approach but requires more restructuring and adds a synchronization point.

### Why Option A is preferred

- Minimal change — only adds one method to `broker_queue.h` and changes one call site in `mustard.h`.
- Semantically correct — the initial queue seed is a purely local, single-PE operation. Using NVSHMEM ops for it was never necessary.
- Does not affect the actual cross-GPU communication, which still goes through NVSHMEM in `kernel_dep_update` and `kernel_scheduler`.
