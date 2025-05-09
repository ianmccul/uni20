#pragma once
#include "common/trace.hpp"
// #include "scheduler.hpp"
#include <atomic>
#include <concepts>
#include <coroutine>
#include <deque>
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
      TRACE("Got some coroutines to resume", HCopy.size());
      std::reverse(HCopy.begin(), HCopy.end()); // reverse the execution order
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
    friend class EpochQueue;
};

/// \brief Asynchronous handle for fire-and-forget coroutines.
struct AsyncTask
{
    struct promise_type
    {
        Scheduler* sched_ = nullptr;

        promise_type() noexcept = default;
        AsyncTask get_return_object() noexcept
        {
          return AsyncTask{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> h_;

    explicit AsyncTask(std::coroutine_handle<promise_type> h) noexcept : h_{h} {}

    ~AsyncTask()
    {
      if (h_) h_.destroy();
    }
};

/// \brief One generation’s context: one writer + N readers.
struct EpochContext
{
    using Handle = std::coroutine_handle<AsyncTask::promise_type>;

    EpochContext(bool writer_already_done) noexcept : writer_done_{writer_already_done}, created_readers_{0} {}

    // — writer side —
    void bind_writer(Handle h) noexcept { writer_handle_ = h; }
    void mark_writer_done() noexcept { writer_done_.store(true, std::memory_order_release); }

    // — reader side —
    void create_reader() noexcept
    {
      // increment reader‐count first
      created_readers_.fetch_add(1, std::memory_order_relaxed);
    }

    void add_reader(Handle h) noexcept
    {
      // stash the handle under lock
      std::lock_guard lk(reader_mtx_);
      reader_handles_.push_back(h);
    }
    void mark_reader_done() noexcept { created_readers_.fetch_sub(1, std::memory_order_acq_rel); }

    // — for the scheduler to pull out the batch —
    std::vector<Handle> take_readers() noexcept
    {
      std::vector<Handle> v;
      std::lock_guard lk(reader_mtx_);
      v.swap(reader_handles_);
      return v;
    }

    // — predicates & writer handle extraction —
    bool writer_has_task() const noexcept { return bool(writer_handle_); }
    bool writer_done() const noexcept { return writer_done_.load(std::memory_order_acquire); }
    bool readers_ready() const noexcept { return writer_done(); }
    bool readers_empty() const noexcept { return created_readers_.load(std::memory_order_acquire) == 0; }

    Handle take_writer() noexcept
    {
      auto h = writer_handle_;
      writer_handle_ = {};
      return h;
    }

  private:
    Handle writer_handle_{};
    std::atomic<bool> writer_done_{false};
    std::atomic<int> created_readers_;

    std::mutex reader_mtx_;
    std::vector<Handle> reader_handles_;
};

/// \brief FIFO queue enforcing writer→readers→next-writer ordering.
class EpochQueue {
  public:
    EpochQueue()
    {
      // queue_.emplace_back(/*writer_already_done=*/true); // sentinel epoch
    }

    EpochContext* new_reader() noexcept
    {
      std::lock_guard lk(mtx_);
      // always attach to back
      if (queue_.empty()) queue_.emplace_back(true);
      TRACE(queue_.size());
      auto* e = &queue_.back();
      e->create_reader();
      return e;
    }

    bool has_pending_writers() const noexcept
    {
      std::lock_guard lk(mtx_);
      return queue_.size() > 1 || (queue_.size() == 1 && !queue_.front().writer_done());
    }

    EpochContext* new_writer() noexcept
    {
      std::lock_guard lk(mtx_);
      queue_.emplace_back(/*writer_already_done=*/false);
      TRACE(queue_.size());
      return &queue_.back();
    }

    /// “Is this epoch now at the head of the queue?”
    bool is_front(const EpochContext* e) const noexcept
    {
      std::lock_guard lk(mtx_);
      TRACE(queue_.size(), e, &queue_.front());
      DEBUG_CHECK(!queue_.empty());
      return e == &queue_.front();
    }

    // Called in WriteBuffer::await_suspend
    // We just bound our writer handle → if we're front, fire it immediately.
    void on_writer_bound(EpochContext* e) noexcept
    {
      std::unique_lock lk(mtx_);
      if (e == &queue_.front())
      {
        TRACE("Scheduling writer");
        DEBUG_CHECK(e->writer_has_task(), "Unexpected: no writer to schedule!");
        auto wh = e->take_writer();
        lk.unlock();
        wh.promise().sched_->schedule(wh);
      }
      else
        TRACE("Writer is not ready to run yet");
    }

    // Called when releasing a write buffer.
    // Mark writer done; if front & no readers pending, skip forward.
    // Otherwise, if front & readers *are* pending, schedule them now.
    void on_writer_done(EpochContext* e) noexcept
    {
      e->mark_writer_done();

      // Grab the lock and see if we are at the current epoch
      {
        std::unique_lock lk(mtx_);

        // If this isn't the front epoch, then we are done
        if (e != &queue_.front()) return;

        auto readers = e->take_readers();
        if (!readers.empty())
        {
          // we have some readers we can schedule, so drop the lock and schedule them
          lk.unlock();
          for (auto rh : readers)
            rh.promise().sched_->schedule(rh);
          return;
          // else: front epoch had no readers → fall through to advance()
        }
        queue_.pop_front(); // keep the lock until here, so we make sure that no readers add to the epoch without our
                            // knowing
      }
      this->advance();
    }

    // Called in ReadBuffer::~ReadBuffer or release()
    // Mark one reader done; if front epoch is now fully drained, pop it and
    // immediately schedule the next writer (or skip further).
    void on_reader_done(EpochContext* e) noexcept
    {
      e->mark_reader_done();
      {
        std::lock_guard lk(mtx_);
        // only pop if *this* front epoch has done its writer and all readers
        if (&queue_.front() != e || !e->writer_done() || !e->readers_empty()) return;

        queue_.pop_front();
      }
      this->advance();
    }

  private:
    void advance() noexcept
    {
      while (true)
      {
        std::unique_lock lk(mtx_);
        if (queue_.empty()) return;
        auto* e = &queue_.front();

        // Phase 1: fire writer if not yet fired
        if (e->writer_has_task())
        {
          // bind_writer happened, so now fire it
          auto wh = e->take_writer();
          lk.unlock();
          wh.promise().sched_->schedule(wh);
          return;
        }

        // Phase 2: fire readers once writer_done
        if (e->readers_ready())
        {
          // take the handles
          auto readers = e->take_readers();
          if (!readers.empty())
          {
            lk.unlock();
            for (auto h : readers)
              h.promise().sched_->schedule(h);
            return;
          }
        }

        // Phase 3: pop when both done
        if (e->writer_done() && e->readers_empty())
        {
          queue_.pop_front();
          continue; // loop to next epoch
        }

        // otherwise no further action
        return;
      }
    }

    mutable std::mutex mtx_;
    std::deque<EpochContext> queue_;
};

template <typename T> class ReadBuffer;
template <typename T> class WriteBuffer;

/// \brief Asynchronous value container.
template <typename T> class Async {
  public:
    Async() noexcept = default;
    Async(T v) noexcept : data_(std::move(v)) {}

    /// \brief Acquire a read gate.
    ReadBuffer<T> GetReadBuffer() noexcept { return ReadBuffer<T>(this, queue_.new_reader()); }

    /// \brief Acquire a write gate.
    WriteBuffer<T> GetWriteBuffer() noexcept { return WriteBuffer<T>(this, queue_.new_writer()); }

    /// \brief Direct access (waiting for all prior ops to complete).
    template <typename Scheduler> T& get_wait(Scheduler& s)
    {
      while (queue_.has_pending_writers())
        s.run();
      return data_;
    }

  private:
    friend class ReadBuffer<T>;
    friend class WriteBuffer<T>;

    T* data() noexcept { return &data_; }

    T data_{};
    EpochQueue queue_;
};

/// \brief RAII awaitable for snapshot-reads of Async<T>.
template <typename T> class ReadBuffer {
  public:
    /// \brief Create a read-gate awaitable.
    ReadBuffer(Async<T>* parent, EpochContext* epoch) noexcept : parent_{parent}, epoch_{epoch} {}

    ReadBuffer(ReadBuffer const&) = delete;
    ReadBuffer& operator=(ReadBuffer const&) = delete;

    ReadBuffer(ReadBuffer&& other) noexcept
        : parent_(other.parent_), epoch_(other.epoch_), sched_(other.sched_), done_(other.done_)
    {
      other.done_ = true; // suppress destructor logic
    }

    ReadBuffer& operator=(ReadBuffer&& other) noexcept
    {
      if (!done_)
      {
        epoch_->mark_reader_done();
        parent_->queue_.on_reader_done(epoch_);
      }
      parent_ = other.parent_;
      epoch_ = other.epoch_;
      sched_ = other.sched_;
      done_ = other.done_;
      other.done_ = true; // suppress destructor logic
      return *this;
    }

    bool await_ready() const noexcept
    {
      DEBUG_CHECK(!done_, "ReadBuffer used after release()!");
      DEBUG_TRACE("ReadBuffer await_ready()", epoch_->readers_ready(), epoch_);
      return epoch_->readers_ready();
    }

    template <typename Promise> void await_suspend(std::coroutine_handle<Promise> h) noexcept
    {
      auto hh = std::coroutine_handle<AsyncTask::promise_type>::from_address(h.address());
      epoch_->add_reader(hh);
      sched_ = hh.promise().sched_;
      TRACE("Suspending ReadBuffer", epoch_);
    }

    T& await_resume() const noexcept
    {
      TRACE("Resuming ReadBuffer", epoch_);
      return *parent_->data();
    }

    void release()
    {
      epoch_->mark_reader_done();
      parent_->queue_.on_reader_done(epoch_);
      done_ = true;
    }

    ~ReadBuffer()
    {
      TRACE("Destroying ReadBuffer", done_);
      if (!done_)
      {
        TRACE(epoch_);
        epoch_->mark_reader_done();
        parent_->queue_.on_reader_done(epoch_);
      }
    }

  private:
    Async<T>* parent_;
    EpochContext* epoch_;
    Scheduler* sched_ = nullptr;
    bool done_ = false;
};

/// \brief RAII awaitable for in-place writes to Async<T>.
template <typename T> class WriteBuffer {
  public:
    /// \brief Create a write-gate awaitable.
    WriteBuffer(Async<T>* parent, EpochContext* epoch) noexcept : parent_{parent}, epoch_{epoch} {}

    WriteBuffer(const WriteBuffer&) = delete;
    WriteBuffer& operator=(const WriteBuffer&) = delete;

    WriteBuffer(WriteBuffer&& other) noexcept
        : parent_(other.parent_), epoch_(other.epoch_), sched_(other.sched_), done_(other.done_)
    {
      other.done_ = true;
    }

    WriteBuffer& operator=(WriteBuffer&& other) noexcept
    {
      if (!done_)
      {
        epoch_->mark_writer_done();
        parent_->queue_.on_writer_done(epoch_);
      }
      parent_ = other.parent_;
      epoch_ = other.epoch_;
      sched_ = other.sched_;
      done_ = other.done_;
      other.done_ = true; // suppress destructor logic
      return *this;
    }

    bool await_ready() const noexcept
    {
      DEBUG_CHECK(!done_, "WriteBuffer used after release()!");
      DEBUG_CHECK(!epoch_->writer_has_task(), "unexpected: double bind!");
      // only ready to run if we are at the front of the queue
      TRACE("WriteBuffer await_ready()", parent_->queue_.is_front(epoch_), epoch_);
      return parent_->queue_.is_front(epoch_);
    }

    template <typename Promise> void await_suspend(std::coroutine_handle<Promise> h) noexcept
    {
      auto hh = std::coroutine_handle<AsyncTask::promise_type>::from_address(h.address());
      epoch_->bind_writer(hh);
      sched_ = hh.promise().sched_;
      parent_->queue_.on_writer_bound(epoch_);
      TRACE("Suspending WriteBuffer", epoch_);
    }

    T& await_resume() const noexcept
    {
      TRACE("Resuming WriteBuffer", epoch_);
      return *parent_->data();
    }

    void release()
    {
      epoch_->mark_writer_done();
      parent_->queue_.on_writer_done(epoch_);
      done_ = true;
    }

    ~WriteBuffer()
    {
      TRACE("Destroying WriteBuffer", done_);
      if (!done_)
      {
        TRACE(epoch_);
        epoch_->mark_writer_done();
        parent_->queue_.on_writer_done(epoch_);
      }
    }

  private:
    Async<T>* parent_;
    EpochContext* epoch_;
    Scheduler* sched_ = nullptr;
    bool done_ = false;
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

template <typename T> struct AsyncAwaiter
{
    Async<T>* a_;
    EpochContext* epoch_;

    explicit AsyncAwaiter(Async<T>& a) : a_(&a), epoch_(a.queue_.new_reader()) {}

    bool await_ready() const noexcept { return epoch_->readers_ready(); }

    void await_suspend(std::coroutine_handle<> h)
    {
      auto hh = std::coroutine_handle<AsyncTask::promise_type>::from_address(h.address());
      epoch_->add_reader(hh);
      // hh.promise().sched_->advance(); // kick the queue if needed
    }

    T& await_resume() const noexcept { return *a_->data(); }

    ~AsyncAwaiter()
    {
      epoch_->mark_reader_done();
      a_->queue_.on_reader_done(epoch_);
    }
};

template <typename T> AsyncAwaiter<T> operator co_await(Async<T>& a) { return AsyncAwaiter<T>(a); }
