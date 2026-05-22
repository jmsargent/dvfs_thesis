#include <cuda_runtime.h>
#include <mpi.h>
#include <nvml.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "frequency_controller.h"

using namespace std::chrono_literals;

static FILE* log_ = nullptr;

static unsigned int clockMHz(nvmlDevice_t dev)
{
    unsigned int clk;
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &clk);
    return clk;
}

static void settle(int pe)
{
    int* d = nullptr;
    cudaSetDevice(pe);
    cudaMalloc(&d, 4);
    cudaFree(d);
    cudaDeviceSynchronize();
    std::this_thread::sleep_for(300ms);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int pe, nPes;
    MPI_Comm_rank(MPI_COMM_WORLD, &pe);
    MPI_Comm_size(MPI_COMM_WORLD, &nPes);

    int testFreq = (argc > 1) ? atoi(argv[1]) : 240;
    std::string outDir = (argc > 2) ? argv[2] : ".";

    std::string logPath = outDir + "/test_freq_pe" + std::to_string(pe) + ".log";
    log_ = fopen(logPath.c_str(), "w");

    fprintf(log_, "PE %d/%d  CUDA_VISIBLE_DEVICES=%s  testFreq=%d\n\n",
            pe, nPes,
            getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "(unset)",
            testFreq);

    cudaSetDevice(pe);

    NvmlFrequencyController ctrl(pe);

    nvmlDevice_t dev = ctrl.device().value();

    unsigned int before = clockMHz(dev);
    fprintf(log_, "clock before setFrequency(%d): %u MHz\n", testFreq, before);

    ctrl.setFrequency(testFreq);
    settle(pe);

    unsigned int after = clockMHz(dev);
    fprintf(log_, "clock after  setFrequency(%d): %u MHz  %s\n", testFreq, after,
            (after != before) ? "ok" : "NO CHANGE");

    ctrl.setFrequency(2040);
    settle(pe);

    fprintf(log_, "clock after  setFrequency(2040): %u MHz\n", clockMHz(dev));

    fclose(log_);
    MPI_Finalize();
    return 0;
}
