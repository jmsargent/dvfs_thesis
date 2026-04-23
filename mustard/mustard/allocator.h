#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "utils.h"

namespace mustard
{

class TaskAllocator
{
   public:
    int* allocate(const std::vector<int>& data)
    {
        if (data.empty()) return nullptr;
        int* ptr;
        checkCudaErrors(cudaMalloc(&ptr, sizeof(int) * data.size()));
        checkCudaErrors(
            cudaMemcpy(ptr, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));
        allocations_.push_back(ptr);
        return ptr;
    }

    ~TaskAllocator()
    {
        for (int* p : allocations_) cudaFree(p);
    }

   private:
    std::vector<int*> allocations_;
};

}  // namespace mustard
