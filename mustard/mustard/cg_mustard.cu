#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "mustard.h"

// Performs a global sum reduction and broadcasts the result to all PEs
__global__ void mustard_allreduce_sum_kernel(float* d_val)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();

        // 1. Accumulate all values to PE 0
        if (mype == 0)
        {
            for (int i = 1; i < npes; i++)
            {
                *d_val += nvshmem_float_g(d_val, i);
            }
        }

        // 2. Global Barrier to ensure PE 0 has the total
        nvshmem_barrier_all();

        // 3. Broadcast the total from PE 0 to all other PEs
        float total = nvshmem_float_g(d_val, 0);
        *d_val      = total;
    }
}

// Performs a global allgather to reconstruct the full search direction vector p
__global__ void mustard_allgather_p_kernel(float* d_p, float* d_r, int local_n)
{
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // Each PE puts its local residual r into the correct slot of every other PE's p
    for (int i = 0; i < npes; i++)
    {
        if (i != mype)
        {
            nvshmem_float_put(d_p + (mype * local_n), d_r, local_n, i);
        }
        else
        {
            // Local copy
            for (int j = 0; j < local_n; j++) d_p[mype * local_n + j] = d_r[j];
        }
    }
    nvshmem_barrier_all();
}

__global__ void scalar_divide_kernel(float* dot_rr, float* dot_pq, float* alpha, float* dot_rr_new)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *alpha = (*dot_rr) / (*dot_pq);
    }
}

__global__ void beta_calc_kernel(float* dot_rr_old, float* dot_rr_new, float* beta)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *beta = (*dot_rr_new) / (*dot_rr_old);
    }
}

__global__ void scalar_negate_kernel(float* alpha, float* neg_alpha)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *neg_alpha = -(*alpha);
    }
}

// file: mustard/cg_mustard.cu
/*
 * Step Size ($\alpha$): Minimizes the error along the current search direction.Residual
 * ($r$): Formally defined as $b - Ax$. In the loop, it is updated recursively to avoid a full
 * matrix-vector multiplication.Search Direction
 * ($p$): Unlike steepest descent, CG uses "Conjugate" directions, meaning $p_i^T A p_j = 0$ for $i
 * \neq j$. This prevents the algorithm from "repeating" work in directions it has already
 * optimized.Correction Factor
 * ($\beta$): Used to modify the residual into a new conjugate search direction.
 */
void solve_mustard_cg_full(cublasHandle_t handle, cudaStream_t stream, float* d_A, float* d_x,
                           float* d_r, float* d_p, float* d_q, float* d_dot_rr, float* d_dot_pq,
                           float* d_dot_rr_new, float* d_alpha, float* d_beta, int n, int max_iter)
{
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSetStream(handle, stream);

    float *d_one, *d_zero, *d_neg_alpha;
    cudaMalloc(&d_one, sizeof(float));
    cudaMalloc(&d_zero, sizeof(float));
    cudaMalloc(&d_neg_alpha, sizeof(float));
    float h_one = 1.0f, h_zero = 0.0f;
    cudaMemcpy(d_one, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zero, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    mustard::TiledGraphCreator creator(stream, graph, false);

    MatrixTile t_A     = {0, 0};
    MatrixTile t_p     = {1, 0};
    MatrixTile t_q     = {2, 0};
    MatrixTile t_alpha = {3, 0};
    MatrixTile t_x     = {4, 0};
    MatrixTile t_r     = {5, 0};
    MatrixTile t_beta  = {6, 0};

    // 1. Compute search direction projection: q = A * p
    // This identifies how the search direction p maps into the space of A.
    creator.beginCaptureOperation(t_q, {t_A, t_p});
    cublasSgemv(handle, CUBLAS_OP_N, n, n, d_one, d_A, n, d_p, 1, d_zero, d_q, 1);
    creator.endCaptureOperation();

    // 2. Compute denominator for step size: dot_pq = p^T * q (which is p^T * A * p)
    creator.beginCaptureOperation(t_alpha, {t_p, t_q});
    cublasSdot(handle, n, d_p, 1, d_q, 1, d_dot_pq);
    creator.endCaptureOperation();

    // 3. Compute step size: alpha = (r^T * r) / (p^T * A * p)
    // alpha determines how far to move along the search direction p.
    creator.beginCaptureOperation(t_alpha, {t_alpha});
    scalar_divide_kernel<<<1, 1, 0, stream>>>(d_dot_rr, d_dot_pq, d_alpha, d_dot_rr_new);
    creator.endCaptureOperation();

    // 4. Update solution vector: x = x + alpha * p
    creator.beginCaptureOperation(t_x, {t_alpha, t_p, t_x});
    cublasSaxpy(handle, n, d_alpha, d_p, 1, d_x, 1);
    creator.endCaptureOperation();

    // 5. Update residual vector: r = r - alpha * q (where q = A * p)
    // The residual r represents the local error: r = b - A*x.
    creator.beginCaptureOperation(t_r, {t_alpha, t_q, t_r});
    scalar_negate_kernel<<<1, 1, 0, stream>>>(d_alpha, d_neg_alpha);
    cublasSaxpy(handle, n, d_neg_alpha, d_q, 1, d_r, 1);
    creator.endCaptureOperation();

    // 6. Compute new squared residual norm: dot_rr_new = r_new^T * r_new
    creator.beginCaptureOperation(t_beta, {t_r});
    cublasSdot(handle, n, d_r, 1, d_r, 1, d_dot_rr_new);
    creator.endCaptureOperation();

    // 7. Compute Graham-Schmidt coefficient: beta = (r_new^T * r_new) / (r_old^T * r_old)
    // beta ensures the new search direction is A-orthogonal to previous directions.
    creator.beginCaptureOperation(t_beta, {t_beta});
    beta_calc_kernel<<<1, 1, 0, stream>>>(d_dot_rr, d_dot_rr_new, d_beta);
    creator.endCaptureOperation();

    // 8. Update search direction: p = r_new + beta * p_old
    // This constructs the next A-orthogonal direction for the next iteration.
    creator.beginCaptureOperation(t_p, {t_beta, t_r, t_p});
    cublasSscal(handle, n, d_beta, d_p, 1);
    cublasSaxpy(handle, n, d_one, d_r, 1, d_p, 1);
    // Cycle the residual norm for the next iteration: r_old_norm = r_new_norm
    cudaMemcpyAsync(d_dot_rr, d_dot_rr_new, sizeof(float), cudaMemcpyDeviceToDevice, stream);
    creator.endCaptureOperation();

    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    for (int i = 0; i < max_iter; i++)
    {
        cudaGraphLaunch(instance, stream);
    }
    cudaStreamSynchronize(stream);

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaFree(d_one);
    cudaFree(d_zero);
    cudaFree(d_neg_alpha);
}

void cg_solver_cluster(cublasHandle_t handle, cudaStream_t stream, float* d_A, float* d_x,
                       float* d_r, float* d_p, float* d_q, float* d_dot_rr, float* d_dot_pq,
                       float* d_dot_rr_new, float* d_alpha, float* d_beta, int n, int max_iter)
{
    int mype    = nvshmem_my_pe();
    int npes    = nvshmem_n_pes();
    int local_n = n / npes;  // Assuming n is divisible by npes for simplicity

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasSetStream(handle, stream);

    // NVSHMEM-allocated scalars for global reductions
    float* d_one       = (float*)nvshmem_malloc(sizeof(float));
    float* d_zero      = (float*)nvshmem_malloc(sizeof(float));
    float* d_neg_alpha = (float*)nvshmem_malloc(sizeof(float));
    float  h_one = 1.0f, h_zero = 0.0f;
    cudaMemcpy(d_one, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zero, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // mustard::TiledGraphCreator uses NVSHMEM-backed dependency tracking
    mustard::TiledGraphCreator creator(stream, graph, false);

    MatrixTile t_A = {0, 0}, t_p = {1, 0}, t_q = {2, 0}, t_alpha = {3, 0}, t_x = {4, 0},
               t_r = {5, 0}, t_beta = {6, 0};

    // 1. Parallel Matrix-Vector: q = A_local * p_global
    // Each PE computes a piece of q using its local slice of A.
    // Note: p_global must be gathered or accessed via NVSHMEM if not replicated.
    creator.beginCaptureOperation(t_q, {t_A, t_p});
    cublasSgemv(handle, CUBLAS_OP_N, local_n, n, d_one, d_A, local_n, d_p, 1, d_zero, d_q, 1);
    creator.endCaptureOperation();

    // 2. Distributed Dot Product: dot_pq = sum(p_local^T * q_local)
    // First, compute local dot product, then perform a global reduction.
    creator.beginCaptureOperation(t_alpha, {t_p, t_q});
    cublasSdot(handle, local_n, d_p + (mype * local_n), 1, d_q, 1, d_dot_pq);
    // MUSTARD DIFFERENCE: Insert a collective reduction node into the graph
    mustard_allreduce_sum_kernel<<<1, 1, 0, stream>>>(d_dot_pq);
    creator.endCaptureOperation();

    // 3. Step size alpha: Global value, computed identically on all PEs
    creator.beginCaptureOperation(t_alpha, {t_alpha});
    scalar_divide_kernel<<<1, 1, 0, stream>>>(d_dot_rr, d_dot_pq, d_alpha, d_dot_rr_new);
    creator.endCaptureOperation();

    // 4. Parallel Update: x_local = x_local + alpha * p_local
    creator.beginCaptureOperation(t_x, {t_alpha, t_p, t_x});
    cublasSaxpy(handle, local_n, d_alpha, d_p + (mype * local_n), 1, d_x + (mype * local_n), 1);
    creator.endCaptureOperation();

    // 5. Parallel Residual: r_local = r_local - alpha * q_local
    creator.beginCaptureOperation(t_r, {t_alpha, t_q, t_r});
    scalar_negate_kernel<<<1, 1, 0, stream>>>(d_alpha, d_neg_alpha);
    cublasSaxpy(handle, local_n, d_neg_alpha, d_q, 1, d_r, 1);
    creator.endCaptureOperation();

    // 6. Global Residual Norm: dot_rr_new = sum(r_local^T * r_local)
    creator.beginCaptureOperation(t_beta, {t_r});
    cublasSdot(handle, local_n, d_r, 1, d_r, 1, d_dot_rr_new);
    mustard_allreduce_sum_kernel<<<1, 1, 0, stream>>>(d_dot_rr_new);
    creator.endCaptureOperation();

    // 7. Global Beta Calculation
    creator.beginCaptureOperation(t_beta, {t_beta});
    beta_calc_kernel<<<1, 1, 0, stream>>>(d_dot_rr, d_dot_rr_new, d_beta);
    creator.endCaptureOperation();

    // 8. Global Search Direction Update: p_global = r_global + beta * p_global
    // If p is replicated, all PEs perform the update on the full vector.
    creator.beginCaptureOperation(t_p, {t_beta, t_r, t_p});
    cublasSscal(handle, n, d_beta, d_p, 1);
    // Since r is partitioned, we need an Allgather here to update the full p
    mustard_allgather_p_kernel<<<1, 1, 0, stream>>>(d_p, d_r, local_n);
    cudaMemcpyAsync(d_dot_rr, d_dot_rr_new, sizeof(float), cudaMemcpyDeviceToDevice, stream);
    creator.endCaptureOperation();

    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

    for (int i = 0; i < max_iter; i++)
    {
        cudaGraphLaunch(instance, stream);
    }
    cudaStreamSynchronize(stream);

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    nvshmem_free(d_one);
    nvshmem_free(d_zero);
    nvshmem_free(d_neg_alpha);
}

// Helper to prepare the initial state for the CG iteration
void initialize_mustard_cg(cublasHandle_t handle, cudaStream_t stream, float* d_A, float* d_b,
                           float* d_x, float* d_r, float* d_p, float* d_dot_rr, int n)
{
    // Set pointer mode to HOST for the setup phase to simplify initialization logic
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetStream(handle, stream);

    float one     = 1.0f;
    float neg_one = -1.0f;
    float zero    = 0.0f;

    // 1. r = b (Copy b to r)
    cudaMemcpyAsync(d_r, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // 2. r = r - A * x_0  => r = -1.0 * A * x + 1.0 * r
    // If x_0 is typically zero, you can skip this and just use r = b
    cublasSgemv(handle, CUBLAS_OP_N, n, n, &neg_one, d_A, n, d_x, 1, &one, d_r, 1);

    // 3. p = r (Initial direction)
    cudaMemcpyAsync(d_p, d_r, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // 4. dot_rr = r . r (Initial squared residual)
    cublasSdot(handle, n, d_r, 1, d_r, 1, d_dot_rr);

    // Switch to DEVICE mode for the graph-captured iterations
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}

void initialize_cg_distributed(cublasHandle_t handle, cudaStream_t stream, float* d_A, float* d_b,
                               float* d_x, float* d_r, float* d_p, float* d_dot_rr, int n,
                               int local_n, int mype, int npes)
{
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    float  h_one = 1.0f, h_neg_one = -1.0f, h_zero = 0.0f;
    float *d_one, *d_neg_one, *d_zero;
    cudaMalloc(&d_one, sizeof(float));
    cudaMalloc(&d_neg_one, sizeof(float));
    cudaMalloc(&d_zero, sizeof(float));
    cudaMemcpy(d_one, &h_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neg_one, &h_neg_one, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zero, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

    // 1. Local Residual: r_local = b_local - (A_local * x_global)
    // Here A_local is [local_n x n]
    cublasSgemv(handle, CUBLAS_OP_N, local_n, n, d_neg_one, d_A, local_n, d_x, 1, d_one, d_r, 1);

    // 2. Global dot product: dot_rr = r_local . r_local summed across PEs
    cublasSdot(handle, local_n, d_r, 1, d_r, 1, d_dot_rr);
    mustard_allreduce_sum_kernel<<<1, 1, 0, stream>>>(d_dot_rr);

    // 3. Initial Direction: allgather r_local into p_global
    mustard_allgather_p_kernel<<<1, 1, 0, stream>>>(d_p, d_r, local_n);

    cudaStreamSynchronize(stream);
    cudaFree(d_one);
    cudaFree(d_neg_one);
    cudaFree(d_zero);
}

// ---------------------------------------------------------
// Example Wrapper for the full process
// ---------------------------------------------------------
void run_cg_solver(float* d_A, float* d_b, float* d_x, int n, int max_iter)
{
    float *d_r, *d_p, *d_q;
    float *d_dot_rr, *d_dot_pq, *d_dot_rr_new, *d_alpha, *d_beta;

    cudaMalloc(&d_r, n * sizeof(float));
    cudaMalloc(&d_p, n * sizeof(float));
    cudaMalloc(&d_q, n * sizeof(float));
    cudaMalloc(&d_dot_rr, sizeof(float));
    cudaMalloc(&d_dot_pq, sizeof(float));
    cudaMalloc(&d_dot_rr_new, sizeof(float));
    cudaMalloc(&d_alpha, sizeof(float));
    cudaMalloc(&d_beta, sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    initialize_mustard_cg(handle, stream, d_A, d_b, d_x, d_r, d_p, d_dot_rr, n);

    solve_mustard_cg_full(handle, stream, d_A, d_x, d_r, d_p, d_q, d_dot_rr, d_dot_pq, d_dot_rr_new,
                          d_alpha, d_beta, n, max_iter);

    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_dot_rr);
    cudaFree(d_dot_pq);
    cudaFree(d_dot_rr_new);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}

void cpu_cg_verify(const float* A, const float* b, float* x, int n, int max_iter, float tol = 1e-6f)
{
    std::vector<float> r(n), p(n), q(n);
    float              dot_rr, dot_rr_new, alpha, beta, dot_pq;

    // r = b - A*x
    for (int i = 0; i < n; i++)
    {
        float ax = 0;
        for (int j = 0; j < n; j++) ax += A[i * n + j] * x[j];
        r[i] = b[i] - ax;
        p[i] = r[i];
    }

    for (int iter = 0; iter < max_iter; iter++)
    {
        dot_rr = 0;
        for (int i = 0; i < n; i++) dot_rr += r[i] * r[i];

        if (std::sqrt(dot_rr) < tol) break;

        // q = A * p
        for (int i = 0; i < n; i++)
        {
            q[i] = 0;
            for (int j = 0; j < n; j++) q[i] += A[i * n + j] * p[i];
        }

        dot_pq = 0;
        for (int i = 0; i < n; i++) dot_pq += p[i] * q[i];

        alpha = dot_rr / dot_pq;

        for (int i = 0; i < n; i++)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }

        dot_rr_new = 0;
        for (int i = 0; i < n; i++) dot_rr_new += r[i] * r[i];

        beta = dot_rr_new / dot_rr;

        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * p[i];
        }
    }
}

int main(int argc, char** argv)
{
    // Match LU benchmark: init first, then set device based on node-local PE
    nvshmem_init();
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // Set device based on local PE to avoid the "Device or resource busy" error
    int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(local_pe);

    int n       = 1024;
    int local_n = n / npes;

    if (mype == 0)
    {
        printf("Launching Multi-GPU CG with N=%d (%d PEs)\n", n, npes);
    }

    // --- 2. Allocation on Symmetric Heap ---
    float* d_A_local = (float*)nvshmem_malloc(local_n * n * sizeof(float));
    float* d_x       = (float*)nvshmem_malloc(n * sizeof(float));
    float* d_r_local = (float*)nvshmem_malloc(local_n * sizeof(float));
    float* d_p       = (float*)nvshmem_malloc(n * sizeof(float));
    float* d_q_local = (float*)nvshmem_malloc(local_n * sizeof(float));
    float* d_b_local = (float*)nvshmem_malloc(local_n * sizeof(float));

    float* d_dot_rr     = (float*)nvshmem_malloc(sizeof(float));
    float* d_dot_pq     = (float*)nvshmem_malloc(sizeof(float));
    float* d_dot_rr_new = (float*)nvshmem_malloc(sizeof(float));
    float* d_alpha      = (float*)nvshmem_malloc(sizeof(float));
    float* d_beta       = (float*)nvshmem_malloc(sizeof(float));

    // Initialize d_x and d_b (simplified for example)
    cudaMemset(d_x, 0, n * sizeof(float));
    cudaMemset(d_b_local, 0, local_n * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- 3. Distributed Initialization ---
    initialize_cg_distributed(handle, stream, d_A_local, d_b_local, d_x, d_r_local, d_p, d_dot_rr,
                              n, local_n, mype, npes);

    // --- 4. Launch Main Solver ---
    int max_iter = 100;
    cg_solver_cluster(handle, stream, d_A_local, d_x, d_r_local, d_p, d_q_local, d_dot_rr, d_dot_pq,
                      d_dot_rr_new, d_alpha, d_beta, n, max_iter);

    // 5. Gather result for verification
    // Each PE has a valid local slice of d_x if we used the local update pattern.
    // If d_x was updated globally on all PEs, any PE can provide it.

    if (mype == 0)
    {
        std::vector<float> h_x_gpu(n);
        std::vector<float> h_x_cpu(n, 0.0f);

        // Copy the final solution from the device symmetric heap to host
        // Assuming d_x was updated globally (replicated) during Saxpy
        cudaMemcpy(h_x_gpu.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost);

        // We need the original A and b on the CPU to verify
        // In a real HPC scenario, you'd reload these or keep a host copy
        std::vector<float> h_A(n * n);
        std::vector<float> h_b(n);

        // Re-generate or retrieve the original A and b used for d_A_local/d_b_local
        // (Ensure seed consistency if using rand())

        std::cout << "Launching Reference CG on CPU..." << std::endl;
        cpu_cg_verify(h_A.data(), h_b.data(), h_x_cpu.data(), n, max_iter);

        // Verify Results
        float max_err = 0;
        for (int i = 0; i < n; i++)
        {
            max_err = std::max(max_err, std::abs(h_x_gpu[i] - h_x_cpu[i]));
        }

        std::cout << "Max Absolute Difference: " << max_err << std::endl;
        if (max_err < 1e-4)
        {
            std::cout << "VERIFICATION SUCCESSFUL" << std::endl;
        }
        else
        {
            std::cout << "VERIFICATION FAILED" << std::endl;
        }

        float norm_x = 0;
        for (int i = 0; i < n; i++)
        {
            norm_x += h_x_gpu[i] * h_x_gpu[i];
        }
        norm_x = std::sqrt(norm_x);

        std::cout << "GPU Solution Norm: " << norm_x << std::endl;

        if (norm_x < 1e-9)
        {
            std::cout << "WARNING: Solution is near zero. Check matrix initialization."
                      << std::endl;
        }
    }

    nvshmem_finalize();
    return 0;
}