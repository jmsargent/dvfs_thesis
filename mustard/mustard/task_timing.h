#pragma once

#include <cuda_runtime.h>

#include <array>
#include <string>
#include <vector>

#include "graph_assembler.h"
#include "pe_writer.h"
#include "time_utils.cuh"
#include "utils.h"

enum Col { WAIT_MS = 0, COMPUTE_MS, START_TS, END_TS, WAIT_START_TS, WAIT_END_TS, NUM_COLS };

class TaskTimingCollector
{
   public:
    using TaskTiming = std::array<double, NUM_COLS>;

    TaskTimingCollector(TimestampBuffers ts, const std::vector<int>& tasks, int runs,
                        const std::string& measureFlags)
        : ts_(ts),
          tasks_(tasks),
          all_timings_(runs, std::vector<TaskTiming>(tasks.size())),
          h_compTs_(tasks.size() * 2),
          h_waitTs_(tasks.size() * 2)
    {
        auto has = [&](const char* f) { return measureFlags.find(f) != std::string::npos; };
        if (has("wait_ms"))       active_cols_.push_back(WAIT_MS);
        if (has("compute_ms"))    active_cols_.push_back(COMPUTE_MS);
        if (has("start_ts"))      active_cols_.push_back(START_TS);
        if (has("end_ts"))        active_cols_.push_back(END_TS);
        if (has("wait_start_ts")) active_cols_.push_back(WAIT_START_TS);
        if (has("wait_end_ts"))   active_cols_.push_back(WAIT_END_TS);
    }

    bool active() const { return !active_cols_.empty(); }

    void collect(int run, const gpu_clock::CalibrationRef& ts_ref)
    {
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

    void collectTimestamps(int run, const gpu_clock::CalibrationRef& ts_ref)
    {
        if (!has(START_TS) && !has(END_TS) && !has(COMPUTE_MS)) return;
        if (!ts_.d_compTs) return;
        checkCudaErrors(cudaMemcpy(h_compTs_.data(), ts_.d_compTs,
                                   sizeof(unsigned long long) * tasks_.size() * 2,
                                   cudaMemcpyDeviceToHost));
        for (int idx = 0; idx < (int)tasks_.size(); idx++)
        {
            TaskTiming& tt = all_timings_[run][idx];
            if (has(START_TS))
                tt[START_TS] = (double)gpu_clock::globaltimer_to_unix_ns(
                    h_compTs_[idx * 2 + 0], ts_ref);
            if (has(END_TS))
                tt[END_TS] = (double)gpu_clock::globaltimer_to_unix_ns(
                    h_compTs_[idx * 2 + 1], ts_ref);
            if (has(COMPUTE_MS))
                tt[COMPUTE_MS] = (h_compTs_[idx * 2 + 1] - h_compTs_[idx * 2 + 0]) / 1e6;
        }
    }

    void collectWaitTimestamps(int run, const gpu_clock::CalibrationRef& ts_ref)
    {
        if (!has(WAIT_START_TS) && !has(WAIT_END_TS) && !has(WAIT_MS)) return;
        if (!ts_.d_waitTs) return;
        checkCudaErrors(cudaMemcpy(h_waitTs_.data(), ts_.d_waitTs,
                                   sizeof(unsigned long long) * tasks_.size() * 2,
                                   cudaMemcpyDeviceToHost));
        for (int idx = 0; idx < (int)tasks_.size(); idx++)
        {
            TaskTiming& tt = all_timings_[run][idx];
            if (has(WAIT_START_TS) && h_waitTs_[idx * 2 + 0] != 0)
                tt[WAIT_START_TS] = (double)gpu_clock::globaltimer_to_unix_ns(
                    h_waitTs_[idx * 2 + 0], ts_ref);
            if (has(WAIT_END_TS) && h_waitTs_[idx * 2 + 1] != 0)
                tt[WAIT_END_TS] = (double)gpu_clock::globaltimer_to_unix_ns(
                    h_waitTs_[idx * 2 + 1], ts_ref);
            if (has(WAIT_MS) && h_waitTs_[idx * 2 + 0] != 0)
                tt[WAIT_MS] = (h_waitTs_[idx * 2 + 1] - h_waitTs_[idx * 2 + 0]) / 1e6;
        }
    }

    bool has(Col col) const
    {
        for (Col c : active_cols_)
            if (c == col) return true;
        return false;
    }

    TimestampBuffers                      ts_;
    const std::vector<int>&               tasks_;
    std::vector<Col>                      active_cols_;
    std::vector<std::vector<TaskTiming>>  all_timings_;
    std::vector<unsigned long long>       h_compTs_;
    std::vector<unsigned long long>       h_waitTs_;
};
