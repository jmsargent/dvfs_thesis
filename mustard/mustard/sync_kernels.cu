#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>

namespace mustard
{

__global__ void kernel_signal_static(int task_id, int* d_completion_flags, int* d_notify_pes,
                                     int n_notify_pes, int debug)
{
    int one = 1;
    if (debug) printf("[signal] task %d: signaling %d PEs\n", task_id, n_notify_pes);
    nvshmem_quiet();
    for (int i = 0; i < n_notify_pes; i++)
    {
        if (debug)
            printf("[signal] task %d -> PE %d flag[%d]\n", task_id, d_notify_pes[i], task_id);
        nvshmem_int_put(&d_completion_flags[task_id], &one, 1, d_notify_pes[i]);
    }
    if (debug) printf("[signal] task %d: done\n", task_id);
}

__global__ void kernel_wait_static(int* d_deps, int n_deps, int* d_completion_flags, int debug)
{
    if (debug) printf("[wait] waiting on %d deps\n", n_deps);
    for (int i = 0; i < n_deps; i++)
    {
        if (debug)
            printf("[wait] polling flag[%d] (currently %d)\n", d_deps[i],
                   d_completion_flags[d_deps[i]]);
        nvshmem_int_wait_until(&d_completion_flags[d_deps[i]], NVSHMEM_CMP_EQ, 1);
        if (debug) printf("[wait] flag[%d] set\n", d_deps[i]);
    }
    if (debug) printf("[wait] all deps satisfied\n");
}

}  // namespace mustard
