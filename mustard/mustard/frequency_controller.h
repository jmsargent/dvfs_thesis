#pragma once

#include <nvml.h>
#include <optional>

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
        nvmlDeviceGetHandleByIndex(pe, &device_);
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
