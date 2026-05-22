#pragma once

#include <cuda_runtime.h>
#include <nvml.h>
#include <optional>
#include <stdexcept>
#include <string>

#include "pe_writer.h"

class IFrequencyController
{
   public:
    virtual ~IFrequencyController()              = default;
    virtual void setFrequency(int freqMhz)       = 0;
    virtual std::optional<nvmlDevice_t> device() const { return std::nullopt; }
};

class NvmlFrequencyController : public IFrequencyController
{
   public:
    explicit NvmlFrequencyController(int pe)
    {
        nvmlInit();
        device_ = cudaDeviceToNvmlDevice(pe);
    }

    ~NvmlFrequencyController()
    {
        nvmlDeviceResetGpuLockedClocks(device_);
        nvmlShutdown();
    }

    void setFrequency(int freqMhz) override
    {
        nvmlDeviceSetGpuLockedClocks(device_, freqMhz, freqMhz);
    }

    std::optional<nvmlDevice_t> device() const override { return device_; }

   private:
    // nvmlDeviceGetHandleByIndex uses absolute physical indices and ignores
    // CUDA_VISIBLE_DEVICES. Use the PCI bus ID to look up the NVML handle for
    // the device that CUDA device `cudaDev` actually runs on.
    static nvmlDevice_t cudaDeviceToNvmlDevice(int cudaDev)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, cudaDev) != cudaSuccess)
            throw std::runtime_error("NvmlFrequencyController: cudaGetDeviceProperties failed");
        char busId[32];
        snprintf(busId, sizeof(busId), "%04x:%02x:%02x.0",
                 prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        nvmlDevice_t dev;
        nvmlReturn_t r = nvmlDeviceGetHandleByPciBusId(busId, &dev);
        if (r != NVML_SUCCESS)
            throw std::runtime_error(
                std::string("NvmlFrequencyController: NVML PCI lookup failed: ") +
                nvmlErrorString(r));
        return dev;
    }

    nvmlDevice_t device_;
};

class MockFrequencyController : public IFrequencyController
{
   public:
    void setFrequency(int) override {}
};

class LoggingFrequencyController : public IFrequencyController
{
   public:
    explicit LoggingFrequencyController(PEWriter out) : out_(std::move(out)) {}

    void setFrequency(int freqMhz) override
    {
        out_.print("retune -> %d MHz\n", freqMhz);
        out_.flush();
    }

   private:
    PEWriter out_;
};
