#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

#include "time_utils.cuh"

#define CHECK(call)                                                             \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                    \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main()
{
    int pass = 0, fail = 0;

    unsigned long long* d_ts;
    CHECK(cudaMalloc(&d_ts, sizeof(unsigned long long) * 4));

    // ------------------------------------------------------------------
    // Test 1: timer is non-zero and increases monotonically
    // ------------------------------------------------------------------
    printf("=== Test 1: monotonicity ===\n");
    {
        unsigned long long h_ts[4];
        for (int i = 0; i < 4; i++)
        {
            gpu_clock::kernel_sample_globaltimer<<<1, 1>>>(d_ts + i);
            CHECK(cudaDeviceSynchronize());
        }
        CHECK(cudaMemcpy(h_ts, d_ts, sizeof(unsigned long long) * 4, cudaMemcpyDeviceToHost));

        bool mono = true;
        for (int i = 1; i < 4; i++)
        {
            if (h_ts[i] <= h_ts[i - 1])
            {
                printf("  FAIL: ts[%d]=%llu <= ts[%d]=%llu\n", i, h_ts[i], i - 1, h_ts[i - 1]);
                mono = false;
            }
        }
        if (mono)
        {
            printf("  PASS: timestamps increase: %llu -> %llu -> %llu -> %llu\n",
                   h_ts[0], h_ts[1], h_ts[2], h_ts[3]);
            pass++;
        }
        else
            fail++;
    }

    // ------------------------------------------------------------------
    // Test 2: elapsed GPU time matches a known host sleep (~100 ms)
    // ------------------------------------------------------------------
    printf("=== Test 2: elapsed time over 100 ms sleep ===\n");
    {
        unsigned long long h_start, h_end;

        gpu_clock::kernel_sample_globaltimer<<<1, 1>>>(d_ts);
        CHECK(cudaDeviceSynchronize());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        gpu_clock::kernel_sample_globaltimer<<<1, 1>>>(d_ts + 1);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(&h_start, d_ts + 0, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&h_end,   d_ts + 1, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        long long gpu_elapsed_ms = (long long)(h_end - h_start) / 1000000LL;
        printf("  GPU elapsed:  %lld ms (expected ~100 ms)\n", gpu_elapsed_ms);

        if (gpu_elapsed_ms >= 90 && gpu_elapsed_ms <= 200)
        {
            printf("  PASS\n");
            pass++;
        }
        else
        {
            printf("  FAIL: out of expected range [90, 200] ms\n");
            fail++;
        }
    }

    // ------------------------------------------------------------------
    // Test 3: calibration maps GPU timestamps to reasonable Unix time
    // ------------------------------------------------------------------
    printf("=== Test 3: calibrated absolute timestamps ===\n");
    {
        auto ref = gpu_clock::calibrate();

        unsigned long long h_ts;
        gpu_clock::kernel_sample_globaltimer<<<1, 1>>>(d_ts);
        CHECK(cudaDeviceSynchronize());
        auto wall = std::chrono::system_clock::now();
        CHECK(cudaMemcpy(&h_ts, d_ts, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        int64_t gpu_abs_ns   = gpu_clock::globaltimer_to_unix_ns(h_ts, ref);
        int64_t wall_unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   wall.time_since_epoch()).count();
        double offset_ms = (gpu_abs_ns - wall_unix_ns) / 1e6;

        gpu_clock::print_globaltimer_timestamp("  GPU  abs", h_ts, ref);
        printf("  Wall abs: %lld.%07lld\n",
               (long long)(wall_unix_ns / 1000000000LL),
               (long long)((wall_unix_ns % 1000000000LL) / 100LL));
        printf("  Offset:   %.3f ms\n", offset_ms);

        // The calibration has a small inherent error (kernel launch + memcpy latency).
        // Accept up to 50 ms drift.
        if (std::fabs(offset_ms) < 50.0)
        {
            printf("  PASS: offset within 50 ms\n");
            pass++;
        }
        else
        {
            printf("  FAIL: offset %.3f ms exceeds 50 ms threshold\n", offset_ms);
            fail++;
        }
    }

    // ------------------------------------------------------------------
    printf("\n%d passed, %d failed\n", pass, fail);
    CHECK(cudaFree(d_ts));
    return fail > 0 ? 1 : 0;
}
