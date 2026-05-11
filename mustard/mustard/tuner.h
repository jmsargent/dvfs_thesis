#pragma once

#include <vector>

#include "frequency_scaler.h"

class RuntimeEventHandler
{
   public:
    RuntimeEventHandler(FrequencyScaler&& scaler, const std::vector<CUDASignal>& signals)
        : scaler_(std::move(scaler)), signals_(signals)
    {}

    void init() { scaler_.init(); }

    const std::vector<std::vector<std::string>>& plannedNames() const
    {
        return scaler_.plannedNames();
    }

    void run()
    {
        if (scaler_.intervalCount() == 0) return;

        for (size_t i = 0; i < scaler_.intervalCount() - 1; ++i)
        {
            while (cudaEventQuery(signals_[i].event) != cudaSuccess);
            scaler_.onSignal(i, KernelStatusUpdate::Waiting);
            while (cudaEventQuery(signals_[i].kernelStartEvent) != cudaSuccess);
            scaler_.onSignal(i, KernelStatusUpdate::Running);
        }
    }

    void reset(const std::vector<cudaStream_t>& streams)
    {
        for (size_t i = 0; i < signals_.size(); ++i)
        {
            cudaEventRecord(signals_[i].event, streams[i % streams.size()]);
            if (signals_[i].kernelStartEvent)
                cudaEventRecord(signals_[i].kernelStartEvent, streams[i % streams.size()]);
        }
    }

   private:
    FrequencyScaler                scaler_;
    const std::vector<CUDASignal>& signals_;
};