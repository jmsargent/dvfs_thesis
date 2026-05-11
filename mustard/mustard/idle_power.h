#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include <nvml.h>

class IdlePower
{
   public:
    virtual ~IdlePower() = default;

    virtual double watts() const = 0;

    // energy_uj for a retune of duration_ns at idle
    double energyUj(double duration_ns) const { return watts() * duration_ns * 1e-6; }

    static std::unique_ptr<IdlePower> forDevice(nvmlDevice_t device);
};

// L4 idle power at minimum frequency (~210 MHz), no active compute
// Source: NVIDIA L4 datasheet / empirical measurement
class L4IdlePower : public IdlePower
{
   public:
    double watts() const override { return 15.0; }
};

// L40S idle power at minimum frequency (~210 MHz), no active compute
// Source: NVIDIA L40S datasheet / empirical measurement
class L40SIdlePower : public IdlePower
{
   public:
    double watts() const override { return 35.0; }
};

inline std::unique_ptr<IdlePower> IdlePower::forDevice(nvmlDevice_t device)
{
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlDeviceGetName(device, name, sizeof(name));
    std::string gpuName(name);
    std::transform(gpuName.begin(), gpuName.end(), gpuName.begin(), ::tolower);

    if (gpuName.find("l40s") != std::string::npos) return std::make_unique<L40SIdlePower>();
    if (gpuName.find("l4")   != std::string::npos) return std::make_unique<L4IdlePower>();

    throw std::runtime_error("IdlePower: unsupported GPU \"" + gpuName + "\"");
}
