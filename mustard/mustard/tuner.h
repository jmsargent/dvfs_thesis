#pragma once

#include <memory>
#include <vector>

#include "frequency_scaler.h"

class RuntimeEventHandler
{
   public:
    RuntimeEventHandler(std::unique_ptr<IRuntimeEventHandle> scaler, const std::vector<CUDASignal>& signals)
        : scaler_(std::move(scaler)), signals_(signals)
    {}

    void init() { scaler_->init(); }

    void run()
    {
        if (signals_.empty()) return;

        for (size_t i = 0; i < signals_.size() - 1; ++i)
        {
            while (cudaEventQuery(signals_[i].event) != cudaSuccess);
            scaler_->onSignal(i, signals_[i].nodeIndex, KernelStatusUpdate::Waiting);
            while (cudaEventQuery(signals_[i].kernelStartEvent) != cudaSuccess);
            scaler_->onSignal(i, signals_[i].nodeIndex, KernelStatusUpdate::Running);
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
    std::unique_ptr<IRuntimeEventHandle> scaler_;
    const std::vector<CUDASignal>&       signals_;
};