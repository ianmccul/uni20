/// \file epoch_context.hpp
/// \brief Manages one “generation” of write/read ordering in an Async<T>
/// \ingroup async_core

#pragma once

#include "async_task_promise.hpp"
#include <atomic>
#include <coroutine>
#include <memory>
#include <mutex>
#include <vector>

namespace uni20::async
{

template <typename T> class Async;
template <typename T> class EpochContextReader;

class EpochQueue;

namespace detail
{
template <typename T> struct AsyncImpl;
template <typename T> using AsyncImplPtr = std::shared_ptr<AsyncImpl<T>>;
} // namespace detail

// This needs a rethink.We want to track if the data has been initialized.That is, if we require_read at some point,
//     then we want to track and see if it was.We need to know this,
//     so that operator+= can detect whether we are adding to it,
//     or assigning a new value.

/// \brief Per-epoch state for an Async<T> value, coordinating one writer and multiple readers.
///
/// \details
/// The EpochContext tracks synchronization state for one generation of an Async<T>.
/// One writer coroutine may be bound, and multiple readers.
/// Writer progress is tracked via three flags:
///
///   - \c writer_task_set_ : set once the writer coroutine is registered.
///   - \c writer_done_     : set when the write gate has been released.
///   - \c writer_required_ : true if readers must be destroyed when no write occurs (e.g., canceled).
///   - \c eptr_            : if writer_done_ && writer_required_ then the writer has finished, but not
///                           written to the buffer. In this case, eptr_ == null indicates that the write
///                           was cancelled, which readers may detect and handle, OR if eptr_ != null
///                           then an exception was thrown and this will be passed on to readers.
///                           eptr_ is not itself atomic, but is fenced by writer_required_
///
/// \note
/// Epochs are constructed in a chain. The previous epoch may pass forward `writer_required_`,
/// which means that the existing value is undetermined/invalid, and the writer is required
/// to co_await on the buffer, or we enter an error state.
/// The `counter_` tracks generation number for debugging.
class EpochContext {
  public:
    EpochContext() = delete;

    /// \brief Construct a new epoch.
    /// \param prev Pointer to the previous epoch (if any).
    /// \param writer_already_done If true, this epoch begins in the "bootstrap" state.
    ///
    /// \post If \p writer_already_done is true, this epoch is considered immediately readable.
    EpochContext(EpochContext const* prev, bool writer_already_done) noexcept
        : eptr_(prev ? prev->eptr_ : nullptr), writer_done_{writer_already_done}, writer_required_(false),
          counter_(prev ? prev->counter_ + 1 : 0)
    {
      TRACE_MODULE(ASYNC, "Creating new forwards epoch", this, counter_);
    }

    /// \brief Construct a reverse-mode epoch, linked to the next one in time.
    /// \param next Pointer to the next epoch (i.e., earlier in forward time).
    /// \param writer_already_done If true, this epoch begins in released state.
    EpochContext(EpochContext const* next, std::integral_constant<bool, true> reverse)
        : writer_done_{false}, writer_required_(true), counter_(next ? next->counter_ - 1 : 0)
    {
      TRACE_MODULE(ASYNC, "Creating new reverse epoch", this, counter_);
    }

    // reader interface
  private:
    template <typename T> friend class EpochContextReader;
    /// \brief Reserve a reader slot for this epoch.
    ///
    /// \note Each call increases the reference count for readers. Must be
    ///       matched with a corresponding call to reader_release().
    void reader_acquire() noexcept
    {
      // DEBUG_TRACE_MODULE(ASYNC, "reader_acquire()", this);
      created_readers_.fetch_add(1, std::memory_order_relaxed);
    }

    /// \brief Signal that one reader has completed.
    ///
    /// \note Decreases the reader reference count. When all readers are released,
    ///       the epoch may be advanced by the queue.
    bool reader_release() noexcept
    {
      // DEBUG_TRACE_MODULE(ASYNC, "reader_release()", this, created_readers_.load(std::memory_order_acquire),
      // reader_handles_.size());
      return created_readers_.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

  public:
    /// \brief Enqueue a reader coroutine to be resumed when the epoch is active.
    ///
    /// \param h The suspended reader task.
    /// \pre Must be called after reader_acquire().
    /// \note Stored coroutines are resumed when the epoch advances.
    void reader_enqueue(AsyncTask&& h)
    {
      std::lock_guard lock(reader_mtx_);
      reader_handles_.push_back(std::move(h));
    }

    /// \brief Are readers allowed to run?
    /// \return true if the writer is done
    bool reader_is_ready() const noexcept { return this->writer_is_done(); }

    /// \brief Returns true if reading from the buffer right now would be an error condition
    bool reader_error() const noexcept
    {
      return this->reader_is_ready() && writer_required_.load(std::memory_order_acquire);
    }

    /// \brief returns the exception pointer set by the writer (nullptr, if no exception)
    std::exception_ptr reader_exception() const noexcept { return eptr_; }

    // EpochQueue interface

    /// \brief Extract all pending reader handles.
    /// \return Vector of reader handles.
    std::vector<AsyncTask> reader_take_tasks() noexcept
    {
      TRACE_MODULE(ASYNC, created_readers_.load(std::memory_order_acquire));
      std::lock_guard lock(reader_mtx_);
      std::vector<AsyncTask> v;
      v.swap(reader_handles_);
      return v;
    }

    /// \brief Check if all reader slots have been released.
    ///
    /// \note Intended for use by EpochQueue only during synchronized advancement.
    ///       Not valid for concurrent polling by external threads.
    ///
    /// \pre Caller must hold exclusive access to the epoch queue.
    bool reader_is_empty() const noexcept { return created_readers_.load(std::memory_order_acquire) == 0; }

    void show()
    {
      DEBUG_TRACE_MODULE(ASYNC, this, created_readers_.load(std::memory_order_acquire), reader_handles_.size(),
                         writer_done_.load(std::memory_order_acquire),
                         writer_task_set_.load(std::memory_order_acquire));
    }

    // Writer interface

    // internal private used only by EpochContextWriter<T>

    /// \brief Acquire the writer role for this epoch.
    void writer_acquire() noexcept
    {
      // DEBUG_TRACE_MODULE(ASYNC, "writer_acquire", this);
      created_writers_.fetch_add(1, std::memory_order_relaxed);
    }

    /// \brief Bind a coroutine to act as the writer.
    /// \param task The coroutine task to register.
    /// \pre Must follow writer_acquire(), and only be called once.
    void writer_bind(AsyncTask&& task) noexcept
    {
      // DEBUG_TRACE_MODULE(ASYNC, "Binding writer", this);
      DEBUG_CHECK(!writer_done_.load(std::memory_order_acquire));
      DEBUG_CHECK(!writer_task_set_.load(std::memory_order_acquire));
      writer_task_ = std::move(task);
      eptr_ = nullptr; // reset the exception pointer
      writer_task_set_.store(true, std::memory_order_release);
    }

    /// \brief Mark the writer as complete, releasing the epoch to readers.
    /// \pre Must follow writer_acquire(), and only be called once.
    bool writer_release() noexcept
    {
      DEBUG_CHECK(!writer_done_.load(std::memory_order_acquire));
      bool done = created_writers_.fetch_sub(1, std::memory_order_acq_rel) == 1;

      if (done)
      {
        /// Mark the writer as done, allowing readers to proceed.
        writer_done_.store(true, std::memory_order_release);
      }
      return done;
    }

    // External Writer interface

    // writer interface

    /// \brief Check if a writer coroutine has been bound.
    /// \return true if a writer task was assigned via await_suspend().
    bool writer_has_task() const noexcept { return writer_task_set_.load(std::memory_order_acquire); }

    /// \brief Check if the writer has released the write gate.
    /// \return true if the epoch is ready for readers.
    bool writer_is_done() const noexcept { return writer_done_.load(std::memory_order_acquire); }

    /// \brief Mark this epoch as requiring write; readers will be destroyed if no write occurs.
    /// \note This sets writer_required_ = true. If writer_release() is called without a prior write,
    ///       all reader coroutines will be destroyed.
    void writer_require() noexcept { writer_required_.store(true, std::memory_order_release); }

    /// \brief Mark this epoch as successfully written.
    void writer_has_written() noexcept { writer_required_.store(false, std::memory_order_release); }

    bool writer_is_required() const noexcept { return writer_required_.load(std::memory_order_acquire); }

    void writer_set_exception(std::exception_ptr e) noexcept
    {
      eptr_ = e;
      writer_required_.store(true, std::memory_order_release);
    }

    /// \brief Transfer ownership of the bound writer coroutine.
    /// \return The bound writer coroutine (may be null if none bound).
    AsyncTask writer_take_task() noexcept
    {
      DEBUG_PRECONDITION(writer_task_set_.load(std::memory_order_acquire));
      return std::move(writer_task_);
    }

    // Informational interface

    // return the epoch counter (generation number)
    uint64_t counter() const noexcept { return counter_; }

  private:
    std::atomic<int> created_readers_{0};
    std::mutex reader_mtx_;
    std::vector<AsyncTask> reader_handles_;
    std::exception_ptr eptr_{nullptr}; /// set by the writer, if we have a current exception to pass on to the readers

    AsyncTask writer_task_;                    ///< Coroutine task (if bound).
    std::atomic<int> created_writers_{0};      ///< number of active writers (normally max 1)
    std::atomic<bool> writer_task_set_{false}; ///< Set if task has been bound.
    std::atomic<bool> writer_done_{false};     ///< Set when writer releases gate.
    std::atomic<bool> writer_required_{false}; ///< Flag that we want ReadBuffers to get dropped if we don't write

  public:
    int64_t counter_ = 0;
};

class buffer_cancelled : public std::exception {
  public:
    const char* what() const noexcept override { return "ReadBuffer was cancelled: no value written"; }
};

/// \brief RAII-scoped representation of a reader's participation in an EpochContext.
///
/// This type is constructed by the EpochQueue and passed to a ReadBuffer<T> instance
/// to track a single reader within an epoch. It allows the reader to register for
/// suspension, test readiness, and upon destruction, notify the queue of completion.
///
/// \note This type is move-only. Reader ownership must be transferred or dropped exactly once.
/// \pre The epoch and parent must remain valid for the lifetime of the reader.
template <typename T> class EpochContextReader {
  public:
    /// \brief Default-constructed inactive reader (no effect).
    EpochContextReader() = default;

    EpochContextReader(EpochContextReader const& other) : parent_(other.parent_), epoch_(other.epoch_)
    {
      if (epoch_) epoch_->reader_acquire();
    }

    EpochContextReader& operator=(EpochContextReader const& other)
    {
      if (this != &other)
      {
        if (other.epoch_) other.epoch_->reader_acquire();
        this->release();
        parent_ = other.parent_;
        epoch_ = other.epoch_;
      }
      return *this;
    }

    /// \brief Construct a new reader handle for a given parent and epoch.
    /// \param parent Pointer to the Async<T> owning the queue and data.
    /// \param epoch Pointer to the epoch being tracked.
    EpochContextReader(detail::AsyncImplPtr<T> const& parent, EpochContext* epoch) noexcept
        : parent_(parent), epoch_(epoch)
    {
      if (epoch_) epoch_->reader_acquire();
    }

    /// \brief Move constructor. Transfers ownership and nulls the source.
    EpochContextReader(EpochContextReader&& other) noexcept : parent_(other.parent_), epoch_(other.epoch_)
    {
      other.epoch_ = nullptr;
    }

    /// \brief Move assignment. Releases prior ownership and transfers.
    EpochContextReader& operator=(EpochContextReader&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        parent_ = other.parent_;
        epoch_ = other.epoch_;
        other.epoch_ = nullptr;
      }
      return *this;
    }

    /// \brief Destructor. Signals epoch reader completion if still active.
    /// \post If owning an epoch, marks reader done and notifies the queue.
    ~EpochContextReader() { this->release(); }

    /// \brief Suspend a coroutine task as a reader of this epoch.
    /// \param t The coroutine task to register.
    void suspend(AsyncTask&& t)
    {
      TRACE_MODULE(ASYNC, "suspend", &t, epoch_);
      DEBUG_PRECONDITION(epoch_);
      parent_->queue_.enqueue_reader(epoch_, std::move(t));
    }

    /// \brief Check whether the reader is ready to resume.
    /// \return True if all prerequisites for this epoch are satisfied.
    bool ready() const noexcept
    {
      DEBUG_PRECONDITION(epoch_, this);
      return epoch_->reader_is_ready();
    }

    /// \brief Access the stored value inside the parent Async<T>.
    /// \return Reference to the T value.
    /// \pre The value must be ready. Should only be called after await_ready() returns true.
    T const& data() const
    {
      DEBUG_PRECONDITION(epoch_, this);
      DEBUG_PRECONDITION(epoch_->reader_is_ready());
      DEBUG_TRACE_MODULE(ASYNC, epoch_, epoch_->reader_error(), epoch_->counter_);
      if (epoch_->reader_error())
      {
        if (auto e = epoch_->reader_exception(); e)
          std::rethrow_exception(e);
        else
          throw buffer_cancelled();
      }
      return parent_->value_;
    }

    T const& data_assert() const
    {
      DEBUG_PRECONDITION(epoch_, this);
      DEBUG_PRECONDITION(epoch_->reader_is_ready());
      if (epoch_->reader_error())
      {
        if (auto e = epoch_->reader_exception(); e)
          std::rethrow_exception(e);
        else
        {
          PANIC("buffer cancelled but not caught");
        }
      }
      return parent_->value_;
    }

    // \brief Optionally retrieves the value stored in this buffer, if available.
    ///
    /// \return A pointer to the value, if it is written, otherwise returns nullptr.
    /// \throws Any exception stored in the writer, if the buffer is in an exception state
    /// \pre The buffer must be ready for reading (i.e., the write gate is closed).
    T const* data_maybe() const
    {
      DEBUG_PRECONDITION(epoch_, this);
      DEBUG_PRECONDITION(epoch_->reader_is_ready());
      if (epoch_->reader_error())
      {
        if (auto e = epoch_->reader_exception(); e)
          std::rethrow_exception(e);
        else
          return nullptr;
      }
      return &parent_->value_;
    }

    /// \brief Optionally retrieves the value stored in this buffer, if available.
    ///
    /// \return A copy of the value if it was written; `std::nullopt` if the buffer was cancelled.
    /// \throws Any exception stored in the writer, if the write failed exceptionally.
    /// \pre The buffer must be ready for reading (i.e., the write gate is closed).
    /// \post If a value is returned, it is a full copy and independent of the internal buffer.
    std::optional<T> data_option() const
    {
      DEBUG_PRECONDITION(epoch_, this);
      DEBUG_PRECONDITION(epoch_->reader_is_ready());
      if (epoch_->reader_error())
      {
        if (auto e = epoch_->reader_exception(); e)
          std::rethrow_exception(e);
        else
          return std::nullopt;
      }
      return parent_->value_;
    }

    /// \brief Check whether this epoch is at the front of the queue.
    /// \return True if the associated epoch is the head of the epoch queue.
    bool is_front() const noexcept
    {
      DEBUG_PRECONDITION(epoch_, this);
      return parent_->queue_.is_front(epoch_);
    }

    void release() noexcept
    {
      if (epoch_)
      {
        // TRACE_MODULE(ASYNC, "EpochContextReader release");
        if (epoch_->reader_release()) parent_->queue_.on_all_readers_released(epoch_);
        epoch_ = nullptr;
      }
    }

    /// \brief Wait for the epoch to become available, and then return a reference to the value
    T const& get_wait() const;

  private:
    detail::AsyncImplPtr<T> parent_;
    EpochContext* epoch_ = nullptr; ///< Epoch currently tracked.
};

/// \brief RAII-scoped representation of a writer’s participation in an epoch.
///
/// Constructed by EpochQueue and passed into a WriteBuffer<T>, this class manages
/// the binding and release of a writer coroutine to a single EpochContext. It
/// ensures that the write gate is completed exactly once, either manually via
/// release() or automatically on destruction.
///
/// \ingroup async_core
template <typename T> class EpochContextWriter {
  public:
    /// \brief Construct an active writer.
    EpochContextWriter(detail::AsyncImplPtr<T> const& parent, EpochContext* epoch) noexcept
        : parent_(parent), epoch_(epoch)
    {
      if (epoch_) epoch_->writer_acquire();
    }

    EpochContextWriter(EpochContextWriter const& other) : parent_(other.parent_), epoch_(other.epoch_)
    {
      if (epoch_) epoch_->writer_acquire();
    }

    EpochContextWriter& operator=(EpochContextWriter const&) = delete;

    EpochContextWriter(EpochContextWriter&& other) noexcept : parent_(other.parent_), epoch_(other.epoch_)
    {
      other.epoch_ = nullptr;
    }

    EpochContextWriter& operator=(EpochContextWriter&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        parent_ = other.parent_;
        epoch_ = other.epoch_;
        other.epoch_ = nullptr;
      }
      return *this;
    }

    ~EpochContextWriter() { this->release(); }

    /// \brief Check whether the writer may proceed immediately.
    /// \return true if the writer is at the front of the queue.
    bool ready() const noexcept
    {
      DEBUG_PRECONDITION(epoch_);
      return parent_->queue_.is_front(epoch_);
    }

    /// \brief Suspend the writer task and submit to the epoch queue.
    /// \param t The coroutine to bind and schedule.
    void suspend(AsyncTask&& t)
    {
      DEBUG_PRECONDITION(epoch_);
      TRACE_MODULE(ASYNC, "suspend", &t, epoch_, epoch_->counter_);
      epoch_->writer_bind(std::move(t));
      parent_->queue_.on_writer_bound(epoch_);
    }

    // Access the stored data.
    T& data() const noexcept
    {
      DEBUG_PRECONDITION(parent_);                          // the parent_ must exist
      DEBUG_PRECONDITION(parent_->queue_.is_front(epoch_)); // we must be at the front of the queue
      DEBUG_PRECONDITION(!epoch_->writer_is_done());        // writer still holds the gate
      epoch_->writer_has_written(); // this is the best we can do, to label the write has (or will) actually occur
      return parent_->value_;
    }

    /// \brief Finalize this write gate, if not already done.
    ///
    /// \note This may be called explicitly or automatically by the destructor.
    ///       It is idempotent and safe to call more than once.
    void release() noexcept
    {
      if (epoch_)
      {
        // TRACE_MODULE(ASYNC, "EpochContextWriter: release");
        if (epoch_->writer_release()) parent_->queue_.on_writer_done(epoch_);
        epoch_ = nullptr;
      }
    }

    void writer_require() noexcept
    {
      DEBUG_PRECONDITION(epoch_);
      epoch_->writer_require();
    }

    /// \brief Wait for the epoch to become available, and then return a prvalue-reference to the value
    T&& move_from_wait() const;

  private:
    detail::AsyncImplPtr<T> parent_;
    EpochContext* epoch_ = nullptr;
};

} // namespace uni20::async
