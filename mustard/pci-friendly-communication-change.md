# PCIe-Friendly Communication Change for Panel Factorization

## Problem

The `--panel` mode for both Cholesky (`tiledCholeskyPanel`) and LU (`tiledLUPanel`) distributes
tiles column-wise in a 1D block-cyclic layout: PE 0 owns columns 0, nPEs, 2·nPEs, …; PE 1 owns
columns 1, nPEs+1, …; and so on. At panel step k, all trailing-update operations (SYRK, GEMM for
Cholesky; TRSM_L, GEMM for LU) on non-owning PEs must read panel column k from PE `k % nPEs`.

In the current implementation, `CholeskyCudaOperations` and `LUCudaOperations` pass the remote
NVSHMEM tile pointer directly to cuBLAS:

```cpp
// e.g. syrk in graph_assembler.h
panels_.otherPanel(k % nPEs_).tile(i, k)  // pointer into remote GPU's NVSHMEM region
```

cuBLAS fetches this B×B tile through fine-grained SM load instructions. Without NVLink, each
128-byte cache-line miss crosses the PCIe bus, incurring 16–21 µs latency per miss. For a
256×256 double tile (512 kB), thousands of cache-line fetches serialize through the PCIe link,
explaining observed kernel wait times in the seconds-to-tens-of-seconds range.

Two effects compound:

1. **Panel serialization**: at step k only PE `k % nPEs` works (POTRF/GETRF + TRSM). All other
   PEs idle during this time. With P GPUs, `(P−1)/P` of each panel step is wasted.
2. **Fine-grained PCIe reads**: warp-level cache-line fetches during the compute kernel are the
   least efficient way to move data over PCIe. DMA transfers (Copy Engine) are far more
   bus-efficient for bulk contiguous data.

## Background: Why 1D Block-Cyclic is Communication-Suboptimal

A **2D block-cyclic** layout (e.g. ScaLAPACK's PDPOTRF on a √P × √P process grid) is
communication-optimal for parallel Cholesky. A **1D column-cyclic** layout broadcasts the entire
panel column (O(N·B) words) from a single source PE to all others, exceeding the theoretical lower
bound on communication by a factor of √P.

For LU, CALU (Communication-Avoiding LU) attains the equivalent lower bound via tournament
pivoting (TSLU), eliminating single-PE panel serialization entirely.

## Proposed Minimal Change

Rather than migrating to 2D block-cyclic (which would require restructuring `StridedDevicePanels`,
the graph builder, and the task assignment), a smaller change replaces fine-grained warp loads with
a single DMA transfer per task.

**Before each cuBLAS call that reads a remote panel tile**, insert a `cudaMemcpy2DAsync` that
copies the tile(s) to a compact local staging buffer. cuBLAS then reads from local GPU memory.

Because `cudaMemcpy2DAsync` is issued inside `cudaStreamBeginCapture`, it is recorded as a
`cudaGraphNodeTypeMemcpy` node in the CUDA Graph and serializes correctly before the cuBLAS node —
no changes to the scheduler or dependency tracking are needed.

### What changes

| File | Change |
|---|---|
| `mustard/graph_assembler.h` | Add `stream_`, `myPE_`, `d_staging_[]` to `CholeskyCudaOperations` and `LUCudaOperations`; add `stageTile()` helper; guard the cuBLAS calls in `syrk`, `gemm` (Cholesky) and `trsm_l`, `gemm` (LU) with a conditional `stageTile()` when `k % nPEs != myPE` |
| `cholesky_mustard.cu:555` | Pass `myPE` to `CholeskyCudaOperations::build()` |
| `lu_mustard.cu:1141` | Pass `myPE` to `LUCudaOperations::build()` |

### Staging buffer design

```
d_staging_[wsIdx_]  →  2 × B × B doubles  (compact, lda = B)
                        ├── [0 .. B²-1]       first tile  (e.g. tile(j,k) for GEMM)
                        └── [B² .. 2B²-1]     second tile (e.g. tile(i,k) for GEMM)
```

One staging slot per compute task (indexed by `wsIdx_`, already incremented per task by
`setWorkspace()`). No two concurrent tasks share a staging slot, so there is no race condition even
when multiple streams run in parallel.

Memory cost: `numMyTasks × 2 × B² × 8` bytes. For B=256 and 100 tasks: ≈ 100 MB — acceptable.

### Example: Cholesky SYRK

```cpp
// Before (remote fine-grained reads during compute):
cublasDsyrk(..., panels_.otherPanel(k % nPEs_).tile(i, k), N_, ...);

// After (one DMA transfer, then local read):
bool remote = (k % nPEs_ != myPE_);
double* src_ik = panels_.otherPanel(k % nPEs_).tile(i, k);
int lda_ik = N_;
if (remote) {
    double* stg = d_staging_[wsIdx_];
    cudaMemcpy2DAsync(stg, B_*sizeof(double),
                      src_ik, N_*sizeof(double),
                      B_*sizeof(double), B_,
                      cudaMemcpyDeviceToDevice, stream_);
    src_ik = stg;
    lda_ik = B_;
}
cublasDsyrk(..., src_ik, lda_ik, ...);
```

The same pattern applies to `gemm` (two tiles) and LU's `trsm_l` and `gemm` (one tile each).

## Expected Impact

PCIe P2P bandwidth via DMA (Copy Engine) approaches the link's theoretical maximum. Fine-grained
warp loads achieve a fraction of this due to per-cache-line latency stalls. This change does
**not** fix the panel serialization idle time — that would require moving to 2D block-cyclic or
CALU. It only removes the additional penalty of doing the inter-PE data transfer inefficiently.

## Limitations and Further Work

- **Panel serialization remains**: the structural idle time from 1D block-cyclic is not addressed.
- **2D block-cyclic migration**: would reduce both communication volume (by √P) and panel idle
  time simultaneously. ScaLAPACK's PDPOTRF/PDGETRF are the reference implementations.
- **NCCL broadcast**: replacing independent P2P reads with an NCCL `ncclBroadcast` after panel
  completion would further improve utilization of the PCIe link (ring algorithm vs. one-to-all).
- **Topology check**: run `nvidia-smi topo -m` to determine whether GPUs share a PCIe root
  complex. Cross-socket P2P traverses QPI/UPI and host memory, halving effective bandwidth.

---

## Sources and What Was Obtained From Each

### [1] PCIe P2P Latency Numbers
**Source**: "Benchmark bandwidth and latency of P2P NVIDIA GPUs (NVLink vs PCI)." GitHub Gist, joshlk.
https://gist.github.com/joshlk/bbb1aca6e70b11d251886baee6423dcb

**Used for**: Concrete latency numbers — PCIe P2P latency 16–21 µs without NVLink, 2.2–2.6 µs
with NVLink. Also the observation that without peer access enabled, the path falls back to a host
memory copy. Unidirectional bandwidth ~11 GB/s (PCIe) vs 200–275 GB/s (NVLink).

---

### [2] PCIe P2P Bandwidth on P100 (NVLink vs PCIe)
**Source**: "Comparing NVLink vs PCI-E with NVIDIA Tesla P100 GPUs." Microway HPC Tech Tips.
https://www.microway.com/hpc-tech-tips/comparing-nvlink-vs-pci-e-nvidia-tesla-p100-gpus-openpower-servers/

**Used for**: Corroboration of the ~10 GB/s unidirectional PCIe P2P figure on real hardware
(P100 GPUs on OpenPOWER). Also confirms the 2–3× bandwidth advantage of NVLink over PCIe.

---

### [3] How PCIe P2P Physically Routes Data (BAR mapping)
**Source**: "Part 2: Inter-GPU Communication with PCIe, NVLink, and BARs." Medium / GPU Kernel
Hacking for Engineers.
https://medium.com/gpu-kernel-hacking-for-engineers/part-2-inter-gpu-communication-with-pcie-nvlink-and-bars-19c20d367f44

**Used for**: Explanation of why fine-grained SM loads over PCIe are inefficient — each load goes
through BAR-mapped address space, one cache line at a time, with full PCIe latency per miss. Also
the distinction between same-root-complex P2P (true bypass) and cross-socket P2P (bounces through
host memory).

---

### [4] Cross-Socket PCIe P2P Bandwidth Penalty
**Source**: "Exploring the Complexities of PCIe Connectivity and Peer-to-Peer Communication."
Exxact Corporation blog.
https://www.exxactcorp.com/blog/HPC/exploring-the-complexities-of-pcie-connectivity-and-peer-to-peer-communication

**Used for**: Quantified bandwidth degradation for cross-socket configurations — same socket via
PCIe ~19 GB/s, cross-socket via QPI ~12 GB/s. Also the description of how data traverses
PCIe → CPU 0 → QPI → CPU 1 → PCIe for two-socket systems without NVLink.

---

### [5] 1D vs 2D Block-Cyclic: Communication Lower Bounds
**Source**: Ballard, G., Demmel, J., Holtz, O., Schwartz, O. "Communication lower bounds and
optimal algorithms for numerical linear algebra." Acta Numerica, 2014.
https://users.wfu.edu/ballard/pdfs/Acta14.pdf

**Also**: "Communication-optimal parallel and sequential Cholesky decomposition." SIAM J. Sci.
Comput., 2010. https://arxiv.org/pdf/0902.2537

**Used for**: The theoretical result that the communication lower bound for parallel Cholesky is
Ω(N²/√P) words, and that 2D block-cyclic Cholesky attains this bound. 1D column-cyclic
oversends by a factor of √P relative to the lower bound. This is the formal backing for why 1D
is suboptimal.

---

### [6] Panel Serialization as a Known Bottleneck of 1D Distributions
**Source**: Demmel, J. "CS267 Lecture 12: Distributed Memory Machines." UC Berkeley.
https://people.eecs.berkeley.edu/~demmel/cs267/lecture12/lecture12.html

**Used for**: The explicit statement that 1D column-cyclic layouts concentrate panel factorization
on a single process, creating a serial bottleneck. Also the efficiency formula showing the O(P/N)
bandwidth penalty for 1D vs O(√P/N) for 2D.

**Also**: ScaLAPACK User's Guide. "The Two-dimensional Block-Cyclic Distribution." Netlib.
https://www.netlib.org/scalapack/slug/node75.html

**Used for**: Confirmation that ScaLAPACK uses 2D block-cyclic specifically to avoid the 1D
serial bottleneck, with parallelism in both panel factorization and trailing update.

---

### [7] CALU: Communication-Optimal LU
**Source**: Demmel, J., Grigori, L., Hoemmen, M., Langou, J. "CALU: A Communication Optimal LU
Factorization Algorithm." EECS Tech Report UCB/EECS-2010-29, 2010.
https://bebop.cs.berkeley.edu/pubs/EECS-2010-29.pdf

**Used for**: CALU/TSLU as the communication-optimal alternative to standard LU for high-latency
interconnects. Measured speedups of 2.3–5.5× on various platforms. Establishes that standard
LU with 1D distribution is not communication-optimal.

---

### [8] NVSHMEM P2P Behavior on PCIe
**Source**: NVIDIA NVSHMEM Documentation, v3.6.5.
https://docs.nvidia.com/nvshmem/api/faq.html
https://docs.nvidia.com/nvshmem/api/introduction.html

**Used for**: Confirmation that `nvshmem_ptr` returns a non-null pointer only when the remote PE
is P2P-accessible (peer access enabled, same root complex or compatible topology). The returned
pointer is a BAR-mapped address — kernel loads to it go over PCIe. Also the note that PCIe
NVSHMEM atomics require InfiniBand or UCX.

---

### [9] cuBLASMp Uses 2D Block-Cyclic + NCCL
**Source**: NVIDIA cuBLASMp Documentation.
https://docs.nvidia.com/cuda/cublasmp/

**Used for**: Confirmation that NVIDIA's own multi-GPU BLAS library uses 2D block-cyclic data
layout and NCCL as the communication backend (ring-based collectives rather than P2P reads).
This validates the direction of the "further work" suggestions.
