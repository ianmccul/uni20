#pragma once
/// \file tbb_numa_scheduler.hpp
/// \brief Scheduler that dispatches work across NUMA-aware TbbScheduler arenas.
/// \ingroup async_core

#include "scheduler.hpp"
#include "tbb_scheduler.hpp"

#include <atomic>
#include <fmt/core.h>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <oneapi/tbb/info.h>

namespace uni20::async
{

/// \brief NUMA-aware scheduler that balances work across per-node TBB arenas.
/// \ingroup async_core
class TbbNumaScheduler final : public IScheduler {
  public:
    /// \brief Construct a scheduler that reflects the system's visible NUMA nodes.
    TbbNumaScheduler() : TbbNumaScheduler(oneapi::tbb::info::numa_nodes()) {}

    /// \brief Construct a scheduler with an explicit set of NUMA nodes.
    /// \param nodes NUMA node identifiers to manage.
    explicit TbbNumaScheduler(std::vector<int> nodes) : numa_nodes_(std::move(nodes))
    {
      if (numa_nodes_.empty())
      {
        numa_nodes_.push_back(-1);
      }

      arenas_.reserve(numa_nodes_.size());
      scheduled_counts_.assign(numa_nodes_.size(), 0);

      for (std::size_t i = 0; i < numa_nodes_.size(); ++i)
      {
        auto node = numa_nodes_[i];
        oneapi::tbb::task_arena::constraints constraints;
        constraints.set_numa_id(node);
        auto scheduler = std::make_unique<TbbScheduler>(constraints);
        node_to_index_.emplace(node, arenas_.size());
        arenas_.push_back(Arena{node, std::move(scheduler)});
      }

      fmt::println("[uni20] TbbNumaScheduler: initialized {} NUMA nodes", arenas_.size());
    }

    /// \brief Schedule a coroutine, choosing an arena based on NUMA preference.
    void schedule(AsyncTask&& task) override
    {
      int target = task.preferred_numa_node().value_or(this->select_next_numa_node());
      this->schedule_on_node(std::move(task), target);
    }

    /// \brief Schedule a coroutine on a specific NUMA node.
    /// \param task Coroutine to dispatch.
    /// \param numa_node Requested NUMA node identifier.
    void schedule(AsyncTask&& task, int numa_node) { this->schedule_on_node(std::move(task), numa_node); }

    /// \brief Pause all managed arenas.
    void pause() override
    {
      for (auto& arena : arenas_)
      {
        arena.scheduler->pause();
      }
    }

    /// \brief Resume all managed arenas.
    void resume() override
    {
      for (auto& arena : arenas_)
      {
        arena.scheduler->resume();
      }
    }

    /// \brief Drain all arenas by waiting for completion of pending work.
    void run_all()
    {
      for (auto& arena : arenas_)
      {
        arena.scheduler->run_all();
      }
    }

    /// \brief Access the NUMA nodes managed by this scheduler.
    /// \return Constant reference to the node identifier list.
    [[nodiscard]] const std::vector<int>& numa_nodes() const noexcept { return numa_nodes_; }

    /// \brief Query how many tasks have been dispatched to a NUMA node.
    /// \param numa_node NUMA node identifier to query.
    /// \return Count of tasks scheduled for that node.
    [[nodiscard]] std::size_t scheduled_count_for(int numa_node) const noexcept
    {
      if (arenas_.empty()) return 0;
      auto idx = this->index_for_node(numa_node);
      std::lock_guard<std::mutex> lock(counts_mutex_);
      return scheduled_counts_[idx];
    }

  protected:
    /// \brief Reschedule a coroutine, honoring any recorded NUMA preference.
    void reschedule(AsyncTask&& task) override
    {
      if (auto preferred = task.preferred_numa_node())
      {
        this->schedule_on_node(std::move(task), *preferred);
      }
      else
      {
        this->schedule_on_node(std::move(task), this->select_next_numa_node());
      }
    }

  private:
    struct Arena
    {
        int numa_node;
        std::unique_ptr<TbbScheduler> scheduler;
    };

    [[nodiscard]] std::size_t index_for_node(int numa_node) const noexcept
    {
      if (auto it = node_to_index_.find(numa_node); it != node_to_index_.end())
      {
        return it->second;
      }
      return 0;
    }

    [[nodiscard]] int select_next_numa_node() noexcept
    {
      auto count = arenas_.empty() ? std::size_t{1} : arenas_.size();
      auto index = next_index_.fetch_add(1, std::memory_order_relaxed) % count;
      return arenas_[index].numa_node;
    }

    void schedule_on_node(AsyncTask&& task, int numa_node)
    {
      if (arenas_.empty()) return;
      auto index = this->index_for_node(numa_node);
      auto& arena = arenas_[index];
      auto actual_node = arena.numa_node;
      task.set_preferred_numa_node(actual_node);
      {
        std::lock_guard<std::mutex> lock(counts_mutex_);
        ++scheduled_counts_[index];
      }
      arena.scheduler->schedule(std::move(task));
    }

    std::vector<int> numa_nodes_;
    std::vector<Arena> arenas_;
    std::unordered_map<int, std::size_t> node_to_index_;
    std::vector<std::size_t> scheduled_counts_;
    mutable std::mutex counts_mutex_;
    std::atomic<std::size_t> next_index_{0};
};

} // namespace uni20::async
