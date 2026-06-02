#pragma once

#include <memory>
#include <optional>
#include <stdexcept>

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
        case ScalerMode::CombinedSlackAware:
            return std::make_unique<CombinedSlackAwareFrequencyScaler>(
                repo, std::move(dag), pe, retuneDelayTracker, cfg.baselineFreq,
                std::move(ctrl), std::move(planOut));

        case ScalerMode::ConstantFrequency:
            return std::make_unique<ScalerConstantFrequency>(cfg.baselineFreq, std::move(ctrl));

        case ScalerMode::SyncPoints:
            return std::make_unique<ScalerSyncPoints>(
                repo, std::move(dag), pe, cfg.baselineFreq, goal, std::move(ctrl));

        default:
            throw std::invalid_argument("unknown ScalerMode");
    }
}
