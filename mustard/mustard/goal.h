#pragma once

#include <cmath>

class DVFSGoal
{
   public:
    virtual double operator()(double executionTime_ns, double energy_uj) const = 0;
    virtual ~DVFSGoal() = default;
};

class EDP : public DVFSGoal
{
   public:
    EDP(unsigned n, unsigned m) : n_(n), m_(m) {}

    double operator()(double executionTime_ns, double energy_uj) const override
    {
        return std::pow(energy_uj, n_) * std::pow(executionTime_ns / 1e9, m_);
    }

   private:
    unsigned n_, m_;
};
