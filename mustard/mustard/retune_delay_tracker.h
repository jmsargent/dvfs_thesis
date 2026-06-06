#pragma once

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <cuda_runtime.h>

using namespace std::chrono_literals;

/*
    This uses model by Jing Chen, DEFT
*/
class RetuneDelayTracker
{
   public:
    virtual ~RetuneDelayTracker() = default;

    virtual std::chrono::nanoseconds getRetuneDelay(int from_mhz, int to_mhz) const = 0;
};

class L4RetuneDelayTracker : public RetuneDelayTracker
{
   public:
    std::chrono::nanoseconds getRetuneDelay(int from_mhz, int to_mhz) const override
    {
        if (from_mhz == to_mhz) return {};

        int delta = std::abs(from_mhz - to_mhz);

        if (from_mhz > to_mhz)
        {
            // Ramp down: multi-tiered, predictable
            if (delta <= 555)        return std::chrono::nanoseconds(930 * delta + 168020);
            else if (delta <= 1208)  return 682910ns;
            else                     return std::chrono::nanoseconds(3290 * delta - 3296400);
        }
        else
        {
            // Ramp up: noisy, median-based
            return 315000ns;
        }
    }
};

class L40SRetuneDelayTracker : public RetuneDelayTracker
{
   public:
    std::chrono::nanoseconds getRetuneDelay(int from_mhz, int to_mhz) const override
    {
        if (from_mhz == to_mhz) return {};

        int delta = std::abs(from_mhz - to_mhz);

        if (from_mhz > to_mhz)
        {
            // Ramp down: high variance, conservative median-based
            if (delta < 550)        return std::chrono::nanoseconds(3060 * delta + 89140);
            else if (delta < 2076)  return 1783190ns;
            else                    return std::chrono::nanoseconds(9590 * (delta - 2076) + 1783190);
        }
        else
        {
            // Ramp up
            if (delta < 570)        return std::chrono::nanoseconds(600 * delta + 267940);
            else if (delta < 1055)  return 610580ns;
            else                    return std::chrono::nanoseconds(610580 + 970 * (delta - 1055));
        }
    }
};

inline std::unique_ptr<RetuneDelayTracker> makeRetuneDelayTracker(int device)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (std::string(prop.name).find("L40") != std::string::npos)
        return std::make_unique<L40SRetuneDelayTracker>();
    return std::make_unique<L4RetuneDelayTracker>();
}
