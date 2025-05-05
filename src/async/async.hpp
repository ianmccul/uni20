#pragma once
#include "common/trace.hpp"
// #include "scheduler.hpp"
#include <atomic>
#include <concepts>
#include <coroutine>
#include <list>
#include <memory>
#include <mutex>

class AsyncTask;
class EpochContext;

struct Scheduler
{
    void schedule(AsyncTask&& H);

    void run()
    {
      std::vector<std::coroutine_handle<>> HCopy;
      std::swap(Handles, HCopy);
      for (auto h : HCopy)
      {
        TRACE("running coroutine...");
        h.resume();
      }
    }

    void run_all()
    {
      TRACE(this->done());
      while (!this->done())
      {
        TRACE("running...");
        this->run();
        TRACE(this->done());
      }
    }

    bool done() { return Handles.empty(); }

    std::vector<std::coroutine_handle<>> Handles;

  private:
    void schedule(std::coroutine_handle<> h_) { Handles.push_back(h_); }

    friend class EpochContext;
};

/// \brief One generation’s context: one writer + N readers.
class EpochContext {
  public:
    // default constructor makes an EpochContext with no writer
    EpochContext() : writer_finished_(true) {}

    explicit EpochContext(bool writer_finished) : writer_finished_(writer_finished) {}

    // Construct an EpochContext with the given write_handle
    explicit EpochContext(std::coroutine_handle<> writer_handle)
        : writer_finished_(false), writer_handle_(writer_handle)
    {}

    /// \brief Called by ReadBuffer ctor to register a new reader.

    void add_reader(Scheduler& sched, std::coroutine_handle<> h) noexcept
    {
      std::unique_lock lk(reader_mtx_);

      // Test the atomic *while* holding the lock,
      // so no writer_done() can slip in between.
      if (!writer_finished_.load(std::memory_order_acquire))
      {
        // writer not done yet ⇒ queue the handle for later
        reader_handles_.push_back(h);
        return;
      }

      // writer already finished ⇒ this reader is now "in flight"
      readers_in_flight_.fetch_add(1, std::memory_order_relaxed);

      // unlock before scheduling so we don't hold the lock in the scheduler
      lk.unlock();
      sched.schedule(h);
    }

    // Indicate that the writer can run, as soon as it is scheduled
    void writer_can_run() { writer_can_run_.store(true, std::memory_order_release); }

    bool is_write_ready() const noexcept { return writer_can_run_.load(std::memory_order_acquire); }

    /// \brief Submit the write_handle to the scheduler
    void run_writer(Scheduler& sched)
    {
      DEBUG_CHECK(writer_can_run_.load());
      DEBUG_CHECK(writer_handle_, "null writer_handle_ in EpochContext!");
      sched.schedule(writer_handle_);
      writer_handle_ = {}; // zero out the handle so that we can't accidentally submit it again
    }

    /// \brief Called by WriteBuffer dtor (when the write is logically done).
    ///        This sets the finished flag and re-schedules all readers.
    void writer_done(Scheduler& sched) noexcept
    {
      writer_finished_.store(true, std::memory_order_release);

      std::vector<std::coroutine_handle<>> to_resume;
      {
        std::lock_guard lk(reader_mtx_);
        to_resume.swap(reader_handles_);
      }

      for (auto h : to_resume)
        sched.schedule(h);
      readers_in_flight_.fetch_add(int(to_resume.size()), std::memory_order_relaxed);
    }

    /// \brief Called by each reader’s final_suspend.
    /// \return true if that was the last reader *and* the writer is already done.
    bool reader_done() noexcept
    {
      DEBUG_CHECK(this->is_writer_finished());
      int remaining = readers_in_flight_.fetch_sub(1, std::memory_order_acq_rel) - 1;
      return remaining == 0 && writer_finished_.load(std::memory_order_acquire);
    }

    /// \brief Has the write-side indicated “ready to go”?
    bool is_writer_finished() const noexcept { return writer_finished_.load(std::memory_order_acquire); }

    bool is_finished() noexcept
    {
      std::unique_lock lk(reader_mtx_);
      return this->is_writer_finished() && readers_in_flight_.load(std::memory_order_acquire) == 0 &&
             reader_handles_.empty();
    }

  private:
    std::atomic<bool> writer_finished_{false};
    std::coroutine_handle<> writer_handle_;
    std::atomic<int> readers_in_flight_{0};
    std::atomic<bool> writer_can_run_{false};
    std::mutex reader_mtx_;
    std::vector<std::coroutine_handle<>> reader_handles_;
};

/// \brief FIFO queue enforcing writer→readers→next writer ordering.
class EpochQueue {
    std::mutex mtx_;
    std::list<EpochContext> queue_;

  public:
    /// \brief Start with an empty “epoch 0”.
    // or start with an empty queue, meaning uninitialized
    EpochQueue()
    {
      // Start with a EpochContext that has finished writing
      queue_.emplace_back();
    }

    /// \brief Called by WriteBuffer ctor to register the writer.
    EpochContext* register_writer(std::coroutine_handle<> wh)
    {
      std::lock_guard lk{mtx_};
      queue_.emplace_back(wh);
      if (queue_.size() == 1) queue_.back().writer_can_run();
      return &queue_.back();
    }

    EpochContext* back()
    {
      std::lock_guard lk{mtx_};
      return &queue_.back();
    }

    EpochContext* new_writer()
    {
      std::lock_guard lk{mtx_};
      queue_.emplace_back(false);
      // see if we can run the writer immediately. We might have EpochContexts at the front of the queue
      // that don't contain any jobs
      if (queue_.front().is_finished())
      {
        queue_.pop_front();
        if (queue_.size() == 1) queue_.front().writer_can_run();
      }
      return &queue_.back();
    }

    /// \brief Called by ReadBuffer ctor to register the writer.
    ///        We need to pass in the Scheduler, because if all write operations are complete then
    //         it will be scheduled immediately
    EpochContext* register_reader(Scheduler& sched, std::coroutine_handle<> h)
    {
      std::lock_guard lk{mtx_};
      queue_.back().add_reader(sched, h);
      return &queue_.back();
    }

    /// \brief When each front reader finishes, maybe advance to next writer.
    void readers_finished(Scheduler& sched)
    {
      EpochContext* Head = nullptr;
      {
        std::lock_guard lk{mtx_};
        if (queue_.size() > 1)
        {
          queue_.pop_front();
          Head = &queue_.front();
        }
      }
      if (Head)
      {
        Head->writer_can_run();
        Head->run_writer(sched);
      }
    }
};

/// \brief Forward declaration of Async.
template <typename T> class Async;

/// \brief RAII awaitable for snapshot‐reads of Async<T>.
template <typename T> class ReadBuffer {
    Async<T>* parent_;
    EpochContext* epoch_;
    bool done_;
    Scheduler* sched_ = nullptr;

  public:
    /// \brief Park as a reader of the current epoch.
    explicit ReadBuffer(Async<T>* p, EpochContext* epoch) noexcept : parent_{p}, epoch_(epoch), done_(true) {}

    void release()
    {
      if (!done_ && epoch_->reader_done()) parent_->readers_finished(*sched_);
      done_ = true;
    }

    bool await_ready()
    {
      TRACE("here");
      return epoch_->is_writer_finished();
    }

    void await_suspend(Scheduler* sched, std::coroutine_handle<> h)
    {
      TRACE("suspending...");
      sched_ = sched;
      epoch_->add_reader(*sched, h);
    }

    T& await_resume() const noexcept { return *parent_->data(); }

    /// \brief When destroyed, signal one reader done.
    ~ReadBuffer()
    {
      if (!done_ && epoch_->reader_done()) parent_->readers_finished(*sched_);
    }

    /// \brief Awaiter for co_await readBuf.
    auto operator co_await() const noexcept
    {
      struct Awaiter
      {
          Async<T>* parent_;
          EpochContext* epoch_;

          bool await_ready() const noexcept { return epoch_->is_writer_finished(); }
          void await_suspend(std::coroutine_handle<> h) noexcept
          {
            done_ = false;
            epoch_->add_reader(*sched_, h);
          }
          T& await_resume() const noexcept { return parent_->data_; }
      };
      return Awaiter{parent_, epoch_};
    }
};

/// \brief RAII awaitable for in‐place writes to Async<T>.
template <typename T> class WriteBuffer {
    Async<T>* parent_;
    EpochContext* epoch_;
    bool done_;
    Scheduler* sched_ = nullptr;

  public:
    /// \brief Register the upcoming writer in the queue.
    explicit WriteBuffer(Async<T>* p, EpochContext* epoch) noexcept : parent_{p}, epoch_{epoch}, done_(false) {}

    /// \brief When destroyed, signal writer done.
    void release()
    {
      if (!done_) epoch_->writer_done(*sched_);
      done_ = true;
    }

    ~WriteBuffer()
    {
      if (!done_) epoch_->writer_done(*sched_);
    }

    bool await_ready() const noexcept
    {
      TRACE("here", epoch_->is_write_ready());
      return epoch_->is_write_ready();
    }

    void await_suspend(Scheduler* sched, std::coroutine_handle<> h)
    {
      sched_ = sched;
      // epoch_->add_writer(sched, h); // nothing to do here - the writer will get submitted via the EpochQueue
    }

    T& await_resume() const noexcept { return *parent_->data(); }

    /// \brief Awaiter for co_await writeBuf.
    auto operator co_await() const noexcept
    {
      struct Awaiter
      {
          Async<T>* parent;

          bool await_ready() const noexcept
          {
            std::lock_guard lk{parent->queue_.mtx_};
            auto& e = parent->queue_.queue_.front();
            return e.readers_in_flight == 0 && e.writer.done();
          }
          void await_suspend(std::coroutine_handle<> h) noexcept { parent_->queue_.register_writer(h); }
          T& await_resume() const noexcept { return &parent->data_; }
      };
      return Awaiter{parent_};
    }
};

template <typename A> struct is_read_buffer : std::false_type
{};
template <typename T> struct is_read_buffer<ReadBuffer<T>> : std::true_type
{};

template <typename A> struct is_write_buffer : std::false_type
{};
template <typename T> struct is_write_buffer<WriteBuffer<T>> : std::true_type
{};

template <typename A>
concept SchedulerAwaitable =
    is_read_buffer<std::remove_cvref_t<A>>::value || is_write_buffer<std::remove_cvref_t<A>>::value;

template <SchedulerAwaitable A> struct SchedulerAwareAwaitable
{
    A& inner;         // e.g. ReadBuffer<T> or WriteBuffer<T>
    Scheduler* sched; // injected via promise.await_transform

    // 1) Just forward the ready check
    bool await_ready() const noexcept
    {
      TRACE("here");
      return inner.await_ready();
    }

    // 2) Intercept the suspension and register with our scheduler
    template <typename Promise> void await_suspend(std::coroutine_handle<Promise> h) noexcept
    {
      inner.await_suspend(sched, h);
    }

    // 3) Forward resume
    decltype(auto) await_resume() noexcept { return inner.await_resume(); }
};

/// \brief Asynchronous handle for in‐place T computations.
template <typename T> class Async {
    friend class ReadBuffer<T>;
    friend class WriteBuffer<T>;

    T data_{};
    EpochQueue queue_{};

    void readers_finished(Scheduler& sched) { queue_.readers_finished(sched); }

    T* data() noexcept { return &data_; }

    T const* data() const noexcept { return &data_; }

  public:
    /// \brief Construct with scheduler; initial data is default‐constructed.
    Async() noexcept {}

    /// \brief Construct from a value.
    Async(T v) noexcept : data_(std::move(v)) {}

    /// \brief Acquire a read gate.
    ReadBuffer<T> GetReadBuffer() noexcept { return ReadBuffer<T>{this, queue_.back()}; }

    /// \brief Acquire a write gate.
    WriteBuffer<T> GetWriteBuffer() noexcept { return WriteBuffer<T>{this, queue_.new_writer()}; }

    T& get_wait() noexcept { return data_; }
    T const& get_wait() const noexcept { return data_; }

    /// \brief Snapshot‐copy ctor: schedules “*this = src.data_”.
    Async(Async const& src) noexcept = delete;
    //  : Async{*src.sched_}
    // {
    //   auto read = src.GetReadBuffer();
    //   auto write = this->GetWriteBuffer();
    //   sched_->schedule([read = std::move(read), write = std::move(write)]() -> Async<void> {
    //     auto in = co_await read;
    //     auto out = co_await write;
    //     *out = *in;
    //     co_return;
    //   });
    // }

    Async& operator=(Async&& src) noexcept
    {
      if (&src != this)
      {
        Async tmp{src};
        std::swap(data_, tmp.data_);
      }
      return *this;
    }

    Async& operator=(Async const& src) noexcept = delete;
    // {
    //   if (&src != this)
    //   {
    //     auto read = src.GetReadBuffer();
    //     auto write = this->GetWriteBuffer();
    //     sched_->schedule([read = std::move(read), write = std::move(write)]() -> Async<void> {
    //       auto in = co_await read;
    //       auto out = co_await write;
    //       *out = *in;
    //       co_return;
    //     });
    //   }
    //   return *this;
    // }

    /// \brief In‐place multiply: schedules an async kernel.
    Async& operator*=(Async const& o) noexcept = delete;
    // {
    //   auto read = o.GetReadBuffer();
    //   auto write = this->GetWriteBuffer();
    //   sched_->schedule([read = std::move(read), write = std::move(write)]() -> Async<void> {
    //     auto in = co_await read;
    //     auto out = co_await write;
    //     *out *= *in;
    //     co_return;
    //   });
    //   return *this;
    // }
};

/// \brief Task type for fire‐&‐forget coroutines that return void.
struct AsyncTask
{
    struct promise_type
    {
        Scheduler* sched_{};

        promise_type(Scheduler* s = nullptr) noexcept : sched_{s} {}

        AsyncTask get_return_object() noexcept
        {
          return AsyncTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        std::suspend_never final_suspend() noexcept { return {}; }

        void return_void() noexcept {}

        void unhandled_exception() { std::terminate(); }

        /// \brief Only allow co_await on our managed awaitables.
        template <typename A>
          requires SchedulerAwaitable<A>
        auto await_transform(A&& a)
        {
          TRACE("here");
          return SchedulerAwareAwaitable<A>{std::forward<A>(a), sched_};
        }
        template <typename A>
          requires(!SchedulerAwaitable<A>)
        void await_transform(A&&) = delete;
    };

    std::coroutine_handle<promise_type> h_;
    explicit AsyncTask(std::coroutine_handle<promise_type> h) noexcept : h_(h) {}
    ~AsyncTask()
    {
      TRACE(bool(h_));
      if (h_) h_.destroy();
    }
};

inline void Scheduler::schedule(AsyncTask&& H)
{
  auto h = std::exchange(H.h_, nullptr);

  // Now h is a std::coroutine_handle<AsyncTask::promise_type>,
  // so you can reach into its promise to inject the Scheduler*:
  h.promise().sched_ = this;

  // And finally enqueue that handle for later resume()
  Handles.push_back(h);
}
