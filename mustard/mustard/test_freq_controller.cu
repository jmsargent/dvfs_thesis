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

static void logClocks(nvmlDevice_t dev)
{
    unsigned int graphics, sm, mem;
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &graphics);
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM,       &sm);
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM,      &mem);
    fprintf(log_, "  graphics=%u MHz  sm=%u MHz  mem=%u MHz\n", graphics, sm, mem);
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

    // Log CUDA device identity
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, pe);
    fprintf(log_, "CUDA device %d: %s  PCI %04x:%02x:%02x\n",
            pe, prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    NvmlFrequencyController ctrl(pe);

    nvmlDevice_t dev = ctrl.device().value();

    // Log NVML device identity (PCI bus ID and name)
    nvmlPciInfo_t pci;
    nvmlDeviceGetPciInfo(dev, &pci);
    char nvmlName[96];
    nvmlDeviceGetName(dev, nvmlName, sizeof(nvmlName));
    fprintf(log_, "NVML device:   %s  PCI %s\n\n", nvmlName, pci.busId);

    fprintf(log_, "before setFrequency(%d):\n", testFreq);
    logClocks(dev);

    try { ctrl.setFrequency(testFreq); }
    catch (const std::exception& e) { fprintf(log_, "setFrequency(%d) threw: %s\n", testFreq, e.what()); }
    settle(pe);

    fprintf(log_, "after setFrequency(%d):\n", testFreq);
    logClocks(dev);

    // also try a mid-range frequency that is definitely in range
    int midFreq = 1500;
    try { ctrl.setFrequency(midFreq); }
    catch (const std::exception& e) { fprintf(log_, "setFrequency(%d) threw: %s\n", midFreq, e.what()); }
    settle(pe);

    fprintf(log_, "after setFrequency(%d):\n", midFreq);
    logClocks(dev);

    try { ctrl.setFrequency(2040); }
    catch (const std::exception& e) { fprintf(log_, "setFrequency(2040) threw: %s\n", e.what()); }
    settle(pe);

    fprintf(log_, "after setFrequency(2040):\n");
    logClocks(dev);

    fclose(log_);
    MPI_Finalize();
    return 0;
}
