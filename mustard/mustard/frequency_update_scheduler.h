#pragma once

#include <nvml.h>

#include <atomic>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

class FrequencyUpdateScheduler
{
   public:
    FrequencyUpdateScheduler(nvmlDevice_t device) : device_(device) {}

    ~FrequencyUpdateScheduler() { cancel(); }

    FrequencyUpdateScheduler(const FrequencyUpdateScheduler&)            = delete;
    FrequencyUpdateScheduler& operator=(const FrequencyUpdateScheduler&) = delete;

    void scheduleFrequencyUpdate(int freqMhz, std::chrono::nanoseconds delay)
    {
        if (delay.count() == 0)
        {
            nvmlDeviceSetGpuLockedClocks(device_, freqMhz, freqMhz);
            return;
        }
        cancel();

        completed_.store(false);
        thread_ = std::thread(
            [this, freqMhz, delay]()
            {
                std::this_thread::sleep_for(delay);
                if (!completed_.load(std::memory_order_relaxed))
                    nvmlDeviceSetGpuLockedClocks(device_, freqMhz, freqMhz);
                completed_.store(true, std::memory_order_release);
            });
    }

    void cancel()
    {
        completed_.store(true);
        if (thread_.joinable()) thread_.join();
    }

    bool didComplete() const { return completed_.load(std::memory_order_acquire); }

   private:
    nvmlDevice_t      device_;
    std::thread       thread_;
    std::atomic<bool> completed_{true};
};
