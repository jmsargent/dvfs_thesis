#pragma once

#include <cmath>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tile_operation.h"

struct TaskProfile
{
    int    frequency_mhz;
    int    energy_uj;
    double execution_time_ns;
    std::vector<long> waittimes_ns;
};

// /data/users/sargent/dvfs_thesis/database/cholesky_database.csv
// /data/users/sargent/dvfs_thesis/database/lu_database.csv
// op_name,pe,freq_mhz,energy_consumed_uj,executiontime_ns,waittime0_ns,waittime1_ns,...
class TaskProfileRepository
{
   public:
    using Algorithm = TileOperation::Algorithm;

    explicit TaskProfileRepository(int pe, Algorithm alg) : pe_(pe), alg_(alg) {}

    void insert(const TileOperation& op, int frequency_mhz, int energy_uj, double execution_time_ns,
                std::vector<long> waittimes_ns = {})
    {
        profiles_[op].push_back({frequency_mhz, energy_uj, execution_time_ns, std::move(waittimes_ns)});
    }

    const std::vector<TaskProfile>* getProfiles(const TileOperation& op) const
    {
        auto it = profiles_.find(op);
        if (it == profiles_.end()) return nullptr;
        return &it->second;
    }

    std::optional<TaskProfile> getProfile(const TileOperation& op, int frequency_mhz) const
    {
        auto* ps = getProfiles(op);
        if (!ps) return std::nullopt;
        for (auto& p : *ps)
            if (p.frequency_mhz == frequency_mhz) return p;
        return std::nullopt;
    }

    void loadFromCSV(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open()) throw std::runtime_error("Cannot open profile CSV: " + path);

        std::string line;
        std::getline(f, line);  // skip header

        while (std::getline(f, line))
        {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::string        op_name, tok;
            int                pe, freq_mhz;
            double             energy_uj, exec_ns;

            std::getline(ss, op_name, ',');
            std::getline(ss, tok, ',');
            pe = std::stoi(tok);
            if (pe != pe_) continue;
            std::getline(ss, tok, ',');
            freq_mhz = std::stoi(tok);
            std::getline(ss, tok, ',');
            energy_uj = std::stod(tok);
            std::getline(ss, tok, ',');
            exec_ns = std::stod(tok);

            std::vector<long> waittimes;
            while (std::getline(ss, tok, ','))
                waittimes.push_back(std::stol(tok));

            insert(TileOperation::fromString(op_name, alg_), freq_mhz, (int)std::round(energy_uj),
                   exec_ns, std::move(waittimes));
        }

        for (auto& [op, profs] : profiles_)
            std::sort(profs.begin(), profs.end(), [](const TaskProfile& a, const TaskProfile& b) {
                return a.frequency_mhz < b.frequency_mhz;
            });
    }

    int size() const { return (int)profiles_.size(); }

   private:
int                                   pe_;
    Algorithm                             alg_;
    std::map<TileOperation, std::vector<TaskProfile>>  profiles_;
};
