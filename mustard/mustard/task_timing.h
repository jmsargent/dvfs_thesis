#pragma once

#include <cuda_runtime.h>

#include <array>
#include <string>
#include <vector>

#include "injectors.h"
#include "pe_writer.h"
#include "time_utils.cuh"
#include "utils.h"

enum Col { WAIT_MS = 0, COMPUTE_MS, START_TS, END_TS, WAIT_START_TS, WAIT_END_TS, NUM_COLS };

class TaskTimingCollector
{
   public:
    using TaskTiming = std::array<double, NUM_COLS>;

    TaskTimingCollector(const mustard::InjectionContext& ctx, const std::vector<int>& tasks,
                        int totalNodes, int runs, bool col_wait_ms, bool col_compute_ms,
                        bool col_start_ts, bool col_end_ts, bool col_wait_start_ts,
                        bool col_wait_end_ts)
        : ctx_(ctx), tasks_(tasks), totalNodes_(totalNodes),
          all_timings_(runs, std::vector<TaskTiming>(tasks.size())),
          h_timestamps_(totalNodes * 2), h_wait_timestamps_(totalNodes * 2)
    {
        if (col_wait_ms) active_cols_.push_back(WAIT_MS);
        if (col_compute_ms) active_cols_.push_back(COMPUTE_MS);
        if (col_start_ts) active_cols_.push_back(START_TS);
        if (col_end_ts) active_cols_.push_back(END_TS);
        if (col_wait_start_ts) active_cols_.push_back(WAIT_START_TS);
        if (col_wait_end_ts) active_cols_.push_back(WAIT_END_TS);
    }

    bool active() const { return !active_cols_.empty(); }

    void collect(int run, cudaEvent_t ev_ref, const gpu_clock::CalibrationRef& ts_ref)
    {
        collectEvents(run, ev_ref);
        collectTimestamps(run, ts_ref);
        collectWaitTimestamps(run, ts_ref);
    }

    void write(const std::string& outputPrefix, int myPE,
               const std::vector<std::string>& opNames) const
    {
        if (!active()) return;
        PEWriter out(outputPrefix, myPE);
        out.print("pe,run,task_id,op_name");
        for (Col col : active_cols_) out.print(",%s", colName(col));
        out.print("\n");
        for (int i = 0; i < (int)all_timings_.size(); i++)
        {
            for (int idx = 0; idx < (int)tasks_.size(); idx++)
            {
                int               task = tasks_[idx];
                const TaskTiming& tt   = all_timings_[i][idx];
                out.print("%d,%d,%d,%s", myPE, i, task, opNames[task].c_str());
                for (Col col : active_cols_) printCol(out, col, tt[col]);
                out.print("\n");
            }
        }
        out.flush();
    }

   private:
    static const char* colName(Col col)
    {
        static const char* names[] = {"wait_ms",      "compute_ms",    "start_ts",
                                      "end_ts",        "wait_start_ts", "wait_end_ts"};
        return names[col];
    }

    static void printCol(PEWriter& out, Col col, double v)
    {
        if (col == WAIT_MS || col == COMPUTE_MS)
            out.print(",%.4f", (float)v);
        else
            out.print(",%lld", (long long)v);
    }

    void collectEvents(int run, cudaEvent_t ev_ref)
    {
        if (!has(WAIT_MS) && !has(COMPUTE_MS)) return;
        for (int idx = 0; idx < (int)tasks_.size(); idx++)
        {
            int         task = tasks_[idx];
            TaskTiming& tt   = all_timings_[run][idx];
            if (has(WAIT_MS))
                tt[WAIT_MS] = (ctx_.task_wait_node[task] != nullptr) ? [&]
                {
                    float t;
                    checkCudaErrors(cudaEventElapsedTime(&t, ev_ref, ctx_.compute_start[task]));
                    return (double)t;
                }()
                                                                     : 0.0;
            if (has(COMPUTE_MS))
            {
                float t;
                checkCudaErrors(cudaEventElapsedTime(&t, ctx_.compute_start[task],
                                                     ctx_.compute_end[task]));
                tt[COMPUTE_MS] = t;
            }
        }
    }

    void collectTimestamps(int run, const gpu_clock::CalibrationRef& ts_ref)
    {
        if (!has(START_TS) && !has(END_TS)) return;
        checkCudaErrors(cudaMemcpy(h_timestamps_.data(), ctx_.d_timestamps,
                                   sizeof(unsigned long long) * totalNodes_ * 2,
                                   cudaMemcpyDeviceToHost));
        for (int idx = 0; idx < (int)tasks_.size(); idx++)
        {
            int         task = tasks_[idx];
            TaskTiming& tt   = all_timings_[run][idx];
            if (has(START_TS))
                tt[START_TS] = (double)gpu_clock::globaltimer_to_unix_ns(
                    h_timestamps_[task * 2 + 0], ts_ref);
            if (has(END_TS))
                tt[END_TS] = (double)gpu_clock::globaltimer_to_unix_ns(
                    h_timestamps_[task * 2 + 1], ts_ref);
        }
    }

    void collectWaitTimestamps(int run, const gpu_clock::CalibrationRef& ts_ref)
    {
        if ((!has(WAIT_START_TS) && !has(WAIT_END_TS)) || !ctx_.d_wait_timestamps) return;
        checkCudaErrors(cudaMemcpy(h_wait_timestamps_.data(), ctx_.d_wait_timestamps,
                                   sizeof(unsigned long long) * totalNodes_ * 2,
                                   cudaMemcpyDeviceToHost));
        for (int idx = 0; idx < (int)tasks_.size(); idx++)
        {
            int         task = tasks_[idx];
            TaskTiming& tt   = all_timings_[run][idx];
            if (has(WAIT_START_TS) && h_wait_timestamps_[task * 2 + 0] != 0)
                tt[WAIT_START_TS] = (double)gpu_clock::globaltimer_to_unix_ns(
                    h_wait_timestamps_[task * 2 + 0], ts_ref);
            if (has(WAIT_END_TS) && h_wait_timestamps_[task * 2 + 1] != 0)
                tt[WAIT_END_TS] = (double)gpu_clock::globaltimer_to_unix_ns(
                    h_wait_timestamps_[task * 2 + 1], ts_ref);
        }
    }

    bool has(Col col) const
    {
        for (Col c : active_cols_)
            if (c == col) return true;
        return false;
    }

    const mustard::InjectionContext& ctx_;
    const std::vector<int>&          tasks_;
    int                              totalNodes_;
    std::vector<Col>                 active_cols_;
    std::vector<std::vector<TaskTiming>> all_timings_;
    std::vector<unsigned long long>      h_timestamps_;
    std::vector<unsigned long long>      h_wait_timestamps_;
};
