/// \file epoch_context.hpp
/// \brief Manages one “generation” of write/read ordering.
/// \ingroup async_core

#pragma once

#include "async_task_promise.hpp"
#include <atomic>
#include <coroutine>
#include <mutex>
#include <vector>

namespace uni20::async
{

/// \brief One generation’s context: one writer + N readers.
/// \ingroup async_core
class EpochContext {
  public:
    using Handle = std::coroutine_handle<AsyncTask::promise_type>;

    /// \brief Construct an epoch context.
    /// \param writer_already_done If true, readers may proceed immediately.
    explicit EpochContext(bool writer_already_done) noexcept : writer_done_{writer_already_done}, created_readers_{0} {}

    /// \brief Bind a writer coroutine to this epoch.
    /// \param h Coroutine handle for the writer.
    void bind_writer(Handle h) noexcept { writer_handle_ = h; }

    /// \brief Mark the writer as done, allowing readers to proceed.
    void mark_writer_done() noexcept { writer_done_.store(true, std::memory_order_release); }

    /// \brief Increment the count of anticipated readers.
    void create_reader() noexcept { created_readers_.fetch_add(1, std::memory_order_relaxed); }

    /// \brief Add a suspended reader coroutine to the queue.
    /// \param h Coroutine handle for the reader.
    void add_reader(Handle h) noexcept
    {
      std::lock_guard lock(reader_mtx_);
      reader_handles_.push_back(h);
    }

    /// \brief Mark one reader as finished.
    void mark_reader_done() noexcept { created_readers_.fetch_sub(1, std::memory_order_acq_rel); }

    /// \brief Extract all pending reader handles.
    /// \return Vector of reader handles.
    std::vector<Handle> take_readers() noexcept
    {
      std::lock_guard lock(reader_mtx_);
      std::vector<Handle> v;
      v.swap(reader_handles_);
      return v;
    }

    /// \brief Check if a writer is bound.
    /// \return true if a writer coroutine is present.
    bool writer_has_task() const noexcept { return static_cast<bool>(writer_handle_); }

    /// \brief Check if the writer has completed.
    /// \return true if writer_done_ is true.
    bool writer_done() const noexcept { return writer_done_.load(std::memory_order_acquire); }

    /// \brief Are readers allowed to run?
    /// \return true if the writer is done.
    bool readers_ready() const noexcept { return writer_done(); }

    /// \brief Check if there are no outstanding readers.
    /// \return true if created_readers_ == 0.
    bool readers_empty() const noexcept { return created_readers_.load(std::memory_order_acquire) == 0; }

    /// \brief Extract the bound writer handle.
    /// \return The writer coroutine handle.
    Handle take_writer() noexcept
    {
      auto h = writer_handle_;
      writer_handle_ = {};
      return h;
    }

  private:
    Handle writer_handle_{};
    std::atomic<bool> writer_done_{false};
    std::atomic<int> created_readers_{0};
    std::mutex reader_mtx_;
    std::vector<Handle> reader_handles_;
};

} // namespace uni20::async
