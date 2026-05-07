#pragma once

#include <cmath>

/*
    This uses model by Jing Chen, DEFT 
*/
class RetuneDelayTracker
{
   public:
    virtual ~RetuneDelayTracker() = default;

    // Returns estimated DVFS transition time in microseconds.
    virtual double getRetuneDelay(int from_mhz, int to_mhz) const = 0;
};

class L4RetuneDelayTracker : public RetuneDelayTracker
{
   public:
    double getRetuneDelay(int from_mhz, int to_mhz) const override
    {
        if (from_mhz == to_mhz) return 0.0;

        int delta = std::abs(from_mhz - to_mhz);

        if (from_mhz > to_mhz)
        {
            // Ramp down: multi-tiered, predictable
            if (delta <= 555)        return 0.93 * delta + 168.02;
            else if (delta <= 1208)  return 682.91;
            else                     return 3.29 * delta - 3296.40;
        }
        else
        {
            // Ramp up: noisy, median-based
            return 315.0;
        }
    }
};

class L40SRetuneDelayTracker : public RetuneDelayTracker
{
   public:
    double getRetuneDelay(int from_mhz, int to_mhz) const override
    {
        if (from_mhz == to_mhz) return 0.0;

        int delta = std::abs(from_mhz - to_mhz);

        if (from_mhz > to_mhz)
        {
            // Ramp down: high variance, conservative median-based
            if (delta < 550)        return 3.06 * delta + 89.14;
            else if (delta < 2076)  return 1783.19;
            else                    return 9.59 * (delta - 2076) + 1783.19;
        }
        else
        {
            // Ramp up
            if (delta < 570)        return 0.60 * delta + 267.94;
            else if (delta < 1055)  return 610.58;
            else                    return 610.58 + 0.97 * (delta - 1055);
        }
    }
};
