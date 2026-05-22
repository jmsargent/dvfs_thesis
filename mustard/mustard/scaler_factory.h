#pragma once

#include <memory>
#include <optional>

#include "cli.h"
#include "frequency_controller.h"
#include "frequency_scaler.h"
#include "goal.h"
#include "partitioned_dag.h"
#include "pe_writer.h"
#include "retune_delay_tracker.h"
#include "task_profile_repository.h"

inline std::unique_ptr<IFrequencyController> makeController(const MustardConfig& cfg, int pe)
{
    if (cfg.fakeTuner)
        return std::make_unique<LoggingFrequencyController>(
            PEWriter(cfg.outputDir, "retune", pe, ".log"));
    return std::make_unique<NvmlFrequencyController>(pe);
}

inline std::unique_ptr<IRuntimeEventHandle> makeScaler(
    const MustardConfig&                  cfg,
    const TaskProfileRepository&          repo,
    PartitionedDag<TileAccess>            dag,
    int                                   pe,
    const DVFSGoal&                       goal,
    const RetuneDelayTracker&             retuneDelayTracker,
    std::unique_ptr<IFrequencyController> ctrl,
    std::optional<PEWriter>               planOut = std::nullopt)
{
    switch (cfg.scalerMode)
    {
        case ScalerMode::GreedyNpiDowntune:
            return std::make_unique<GreedyNpiDowntuner>(
                repo, std::move(dag), pe, goal, retuneDelayTracker, cfg.baselineFreq,
                std::move(ctrl), std::move(planOut));

        case ScalerMode::CriticalPathRampUp:
            return std::make_unique<CriticalPathRampUpScaler>(
                repo, std::move(dag), pe, retuneDelayTracker, cfg.baselineFreq,
                std::move(ctrl), std::move(planOut));

        case ScalerMode::NpiGap:
            return std::make_unique<NpiGapScaler>(
                repo, std::move(dag), pe, retuneDelayTracker, cfg.baselineFreq,
                std::move(ctrl), std::move(planOut));

        case ScalerMode::NpiGapRamp:
            return std::make_unique<NpiGapRampScaler>(
                repo, std::move(dag), pe, retuneDelayTracker, cfg.baselineFreq,
                std::move(ctrl), std::move(planOut));

        case ScalerMode::CombinedSlackAware:
            return std::make_unique<CombinedSlackAwareFrequencyScaler>(
                repo, std::move(dag), pe, retuneDelayTracker, cfg.baselineFreq,
                std::move(ctrl), std::move(planOut));
    }
    return nullptr;
}
