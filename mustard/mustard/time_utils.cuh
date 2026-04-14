#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <cuda_runtime.h>

// Print a pre-captured system_clock time_point as a Unix timestamp.
// label     - descriptive string printed before the timestamp
// precision - number of decimal places (1–9; capped at nanoseconds)
inline void print_timestamp(const std::string &label,
                            std::chrono::system_clock::time_point tp,
                            int precision = 7)
{
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();

    int64_t frac_ns = ns % 1000000000LL;
    int     shift   = 9 - precision;
    int64_t frac    = frac_ns;
    for (int i = 0; i < shift; i++) frac /= 10;

    printf("%s: %lld.%0*lld\n", label.c_str(),
           (long long)(ns / 1000000000LL),
           precision,
           (long long)frac);
}

// Print the current wall-clock time as a Unix timestamp.
// label     - descriptive string printed before the timestamp
// precision - number of decimal places (1–9; capped at nanoseconds)
inline void print_timestamp(const std::string &label, int precision = 7)
{
    print_timestamp(label, std::chrono::system_clock::now(), precision);
}

// ---- GPU timer calibration -----------------------------------------------
// Used to convert raw __globaltimer() nanoseconds (from TimestampDecorator)
// into absolute Unix timestamps.
//
// Record a (globaltimer, system_clock) reference pair before launching tasks,
// then call globaltimer_to_unix_ns() per timestamp value.

namespace gpu_clock {

// Small probe kernel: writes __globaltimer() to *out.
static __global__ void kernel_sample_globaltimer(unsigned long long* out)
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *out = t;
}

struct CalibrationRef
{
    unsigned long long globaltimer_ns;  // raw __globaltimer() value at reference point
    int64_t            unix_ns;         // corresponding Unix nanoseconds
};

// Record a reference pair on the given stream (synchronous: blocks until done).
// Call this immediately before launching task graphs to minimise drift.
inline CalibrationRef calibrate(cudaStream_t stream = 0)
{
    unsigned long long* d_buf;
    unsigned long long  h_val;
    cudaMalloc(&d_buf, sizeof(unsigned long long));
    kernel_sample_globaltimer<<<1, 1, 0, stream>>>(d_buf);
    cudaStreamSynchronize(stream);
    auto tp = std::chrono::system_clock::now();
    cudaMemcpyAsync(&h_val, d_buf, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_buf);

    CalibrationRef ref;
    ref.globaltimer_ns = h_val;
    ref.unix_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
    return ref;
}

// Convert a raw __globaltimer() value to Unix nanoseconds using a calibration reference.
inline int64_t globaltimer_to_unix_ns(unsigned long long ts, const CalibrationRef& ref)
{
    return ref.unix_ns + (int64_t)(ts - ref.globaltimer_ns);
}

// Print a raw __globaltimer() value as a Unix timestamp string.
inline void print_globaltimer_timestamp(const std::string& label, unsigned long long ts,
                                        const CalibrationRef& ref, int precision = 7)
{
    int64_t unix_ns = globaltimer_to_unix_ns(ts, ref);
    int64_t frac_ns = unix_ns % 1000000000LL;
    if (frac_ns < 0) frac_ns += 1000000000LL;
    int     shift = 9 - precision;
    int64_t frac  = frac_ns;
    for (int i = 0; i < shift; i++) frac /= 10;
    printf("%s: %lld.%0*lld\n", label.c_str(),
           (long long)(unix_ns / 1000000000LL), precision, (long long)frac);
}

}  // namespace gpu_clock
