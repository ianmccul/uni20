#include <uni20/backend/cuda/runtime.hpp>

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/common/trace.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <ranges>
#include <utility>
#include <vector>

namespace uni20::cuda
{

namespace
{

[[noreturn]] void cleanup_failure(cudaError_t status, char const* operation, int device)
{
  PANIC("CUDA cleanup operation failed", operation, device, cudaGetErrorName(status), cudaGetErrorString(status));
}

void check_cleanup(cudaError_t status, char const* operation, int device) noexcept
{
  if (status != cudaSuccess)
  {
    cleanup_failure(status, operation, device);
  }
}

class CleanupDeviceGuard {
  public:
    explicit CleanupDeviceGuard(int device) noexcept : device_(device)
    {
      check_cleanup(cudaGetDevice(&previous_device_), "cudaGetDevice", device_);
      if (previous_device_ != device_)
      {
        check_cleanup(cudaSetDevice(device_), "cudaSetDevice", device_);
        restore_ = true;
      }
    }

    CleanupDeviceGuard(CleanupDeviceGuard const&) = delete;
    CleanupDeviceGuard& operator=(CleanupDeviceGuard const&) = delete;

    ~CleanupDeviceGuard()
    {
      if (restore_)
      {
        check_cleanup(cudaSetDevice(previous_device_), "cudaSetDevice restore", previous_device_);
      }
    }

  private:
    int device_;
    int previous_device_ = -1;
    bool restore_ = false;
};

} // namespace

ScopedDevice::ScopedDevice(int device) : device_(device), previous_device_(-1), restore_(false)
{
  CHECK(device_ >= 0, device_);
  check(cudaGetDevice(&previous_device_), "cudaGetDevice", device_);
  if (previous_device_ != device_)
  {
    check(cudaSetDevice(device_), "cudaSetDevice", device_);
    restore_ = true;
  }
}

ScopedDevice::~ScopedDevice()
{
  if (restore_)
  {
    check_cleanup(cudaSetDevice(previous_device_), "cudaSetDevice restore", previous_device_);
  }
}

class Completion::State {
  public:
    explicit State(int device) : device_(device)
    {
      ScopedDevice guard(device_);
      check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "cudaEventCreateWithFlags", device_);
    }

    State(State const&) = delete;
    State& operator=(State const&) = delete;

    ~State()
    {
      CleanupDeviceGuard guard(device_);
      check_cleanup(cudaEventDestroy(event_), "cudaEventDestroy", device_);
    }

    [[nodiscard]] int device() const noexcept { return device_; }
    [[nodiscard]] cudaEvent_t native_handle() const noexcept { return event_; }

    void record(int stream_device, cudaStream_t stream) const
    {
      CHECK_EQUAL(device_, stream_device);
      ScopedDevice guard(device_);
      check(cudaEventRecord(event_, stream), "cudaEventRecord", device_);
    }

    [[nodiscard]] bool ready() const
    {
      ScopedDevice guard(device_);
      cudaError_t const status = cudaEventQuery(event_);
      if (status == cudaErrorNotReady)
      {
        return false;
      }
      check(status, "cudaEventQuery", device_);
      return true;
    }

    void synchronize() const
    {
      ScopedDevice guard(device_);
      check(cudaEventSynchronize(event_), "cudaEventSynchronize", device_);
    }

  private:
    int device_;
    cudaEvent_t event_ = nullptr;
};

namespace
{

class StreamResource {
  public:
    StreamResource(int device, unsigned int flags) : device_(device)
    {
      ScopedDevice guard(device_);
      check(cudaStreamCreateWithFlags(&stream_, flags), "cudaStreamCreateWithFlags", device_);
    }

    StreamResource(StreamResource const&) = delete;
    StreamResource& operator=(StreamResource const&) = delete;
    StreamResource(StreamResource&& other) noexcept : device_(other.device_), stream_(other.stream_)
    {
      other.device_ = -1;
      other.stream_ = nullptr;
    }

    StreamResource& operator=(StreamResource&& other) noexcept
    {
      if (this != &other)
      {
        this->reset();
        device_ = other.device_;
        stream_ = other.stream_;
        other.device_ = -1;
        other.stream_ = nullptr;
      }
      return *this;
    }

    ~StreamResource() { this->reset(); }

    [[nodiscard]] int device() const noexcept { return device_; }
    [[nodiscard]] cudaStream_t native_handle() const noexcept { return stream_; }

    void synchronize() const
    {
      CHECK(stream_ != nullptr);
      ScopedDevice guard(device_);
      check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", device_);
    }

  private:
    int device_ = -1;
    cudaStream_t stream_ = nullptr;

    void reset() noexcept
    {
      if (stream_ == nullptr)
      {
        return;
      }
      CleanupDeviceGuard guard(device_);
      check_cleanup(cudaStreamDestroy(stream_), "cudaStreamDestroy", device_);
      device_ = -1;
      stream_ = nullptr;
    }
};

} // namespace

class Stream::State {
  public:
    State(StreamPool::Impl& pool, std::size_t slot) noexcept : pool_(&pool), slot_(slot) {}
    State(State const&) = delete;
    State& operator=(State const&) = delete;
    ~State();

    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] cudaStream_t native_handle() const noexcept;
    void synchronize() const;
    [[nodiscard]] Completion record_completion() const;
    void wait_on(Completion const& completion) const;

  private:
    StreamPool::Impl* pool_;
    std::size_t slot_;
};

Stream::Stream(std::shared_ptr<State> state) noexcept : state_(std::move(state)) {}

int Stream::device() const noexcept
{
  CHECK(state_ != nullptr);
  return state_->device();
}

cudaStream_t Stream::native_handle() const noexcept
{
  CHECK(state_ != nullptr);
  return state_->native_handle();
}

void Stream::synchronize() const
{
  CHECK(state_ != nullptr);
  state_->synchronize();
}

Completion Stream::record_completion() const
{
  CHECK(state_ != nullptr);
  return state_->record_completion();
}

void Stream::wait_on(Completion const& completion) const
{
  CHECK(state_ != nullptr);
  state_->wait_on(completion);
}

Completion::Completion(std::shared_ptr<State> state) : state_(std::move(state)) {}

int Completion::device() const noexcept { return state_ == nullptr ? -1 : state_->device(); }

bool Completion::ready() const
{
  CHECK(state_ != nullptr);
  return state_->ready();
}

void Completion::synchronize() const
{
  CHECK(state_ != nullptr);
  state_->synchronize();
}

class StreamPool::Impl {
  public:
    enum class SlotState
    {
      idle,
      leased,
      pending
    };

    struct Slot
    {
        explicit Slot(int device, unsigned int flags) : stream(device, flags) {}

        StreamResource stream;
        SlotState state = SlotState::idle;
    };

    struct ReturnPayload
    {
        Impl* pool;
        std::size_t slot;
    };

    explicit Impl(Config config) : device_(config.device)
    {
      CHECK(config.device >= 0, config.device);
      CHECK(config.stream_count > 0, config.stream_count);

      int device_count = 0;
      check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
      CHECK(config.device < device_count, config.device, device_count);

      slots_.reserve(config.stream_count);
      for (std::size_t i = 0; i < config.stream_count; ++i)
      {
        slots_.emplace_back(config.device, config.stream_flags);
      }
    }

    ~Impl()
    {
      {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        available_.notify_all();
        CHECK(waiter_head_ == nullptr, "destroying CUDA stream pool with queued waiters");
        CHECK(std::ranges::none_of(slots_, [](Slot const& slot) { return slot.state == SlotState::leased; }),
              "destroying CUDA stream pool with active leases");
      }

      for (auto const& slot : slots_)
      {
        CleanupDeviceGuard guard(device_);
        check_cleanup(cudaStreamSynchronize(slot.stream.native_handle()), "cudaStreamSynchronize", device_);
      }

      std::lock_guard lock(mutex_);
      CHECK(std::ranges::all_of(slots_, [](Slot const& slot) { return slot.state == SlotState::idle; }),
            "CUDA stream pool callback did not return every stream");
    }

    [[nodiscard]] std::optional<std::size_t> try_acquire()
    {
      std::lock_guard lock(mutex_);
      CHECK(!stopping_);
      auto const found = std::ranges::find_if(slots_, [](Slot const& slot) { return slot.state == SlotState::idle; });
      if (found == slots_.end())
      {
        return std::nullopt;
      }
      found->state = SlotState::leased;
      return static_cast<std::size_t>(std::distance(slots_.begin(), found));
    }

    [[nodiscard]] std::size_t acquire()
    {
      std::unique_lock lock(mutex_);
      available_.wait(lock, [this] {
        return stopping_ || std::ranges::any_of(slots_, [](Slot const& slot) { return slot.state == SlotState::idle; });
      });
      CHECK(!stopping_);

      auto const found = std::ranges::find_if(slots_, [](Slot const& slot) { return slot.state == SlotState::idle; });
      CHECK(found != slots_.end());
      found->state = SlotState::leased;
      return static_cast<std::size_t>(std::distance(slots_.begin(), found));
    }

    [[nodiscard]] std::optional<std::size_t> acquire_or_enqueue(detail::StreamWaiter& waiter) noexcept
    {
      std::lock_guard lock(mutex_);
      CHECK(!stopping_);
      CHECK(!waiter.queued_);

      auto const found = std::ranges::find_if(slots_, [](Slot const& slot) { return slot.state == SlotState::idle; });
      if (found != slots_.end())
      {
        found->state = SlotState::leased;
        return static_cast<std::size_t>(std::distance(slots_.begin(), found));
      }

      waiter.queued_ = true;
      waiter.next_ = nullptr;
      if (waiter_tail_ == nullptr)
      {
        waiter_head_ = &waiter;
      }
      else
      {
        waiter_tail_->next_ = &waiter;
      }
      waiter_tail_ = &waiter;
      return std::nullopt;
    }

    [[nodiscard]] StreamResource& stream(std::size_t slot)
    {
      CHECK(slot < slots_.size(), slot, slots_.size());
      return slots_[slot].stream;
    }

    void release_when_idle(std::size_t slot) noexcept
    {
      StreamResource& selected = this->stream(slot);
      auto* payload = new (std::nothrow) ReturnPayload{this, slot};
      if (payload == nullptr)
      {
        this->synchronize_and_mark_idle(slot, selected);
        PANIC("failed to allocate CUDA stream-pool return callback payload", device_, slot);
      }

      {
        std::lock_guard lock(mutex_);
        CHECK(slots_[slot].state == SlotState::leased, slot);
        slots_[slot].state = SlotState::pending;
      }

      // The host function is the stream-pool admission boundary. The stream is
      // not leasable again until all preceding CUDA work and buffer-completion
      // events have completed.
      ScopedDevice guard(device_);
      cudaError_t const status = cudaLaunchHostFunc(selected.native_handle(), &Impl::return_stream_callback, payload);
      if (status != cudaSuccess)
      {
        delete payload;
        this->synchronize_and_mark_idle(slot, selected);
        check_cleanup(status, "cudaLaunchHostFunc stream-pool return", device_);
      }
    }

    void release_leased_without_work(std::size_t slot) noexcept { this->make_available(slot, SlotState::leased); }

    [[nodiscard]] std::size_t count(SlotState state) const noexcept
    {
      std::lock_guard lock(mutex_);
      return static_cast<std::size_t>(
          std::ranges::count_if(slots_, [state](Slot const& slot) { return slot.state == state; }));
    }

    void synchronize()
    {
      for (auto const& slot : slots_)
      {
        slot.stream.synchronize();
      }

      std::lock_guard lock(mutex_);
      CHECK(std::ranges::none_of(slots_, [](Slot const& slot) { return slot.state == SlotState::pending; }),
            "cudaStreamSynchronize returned before a stream-pool host function completed");
    }

    [[nodiscard]] int device() const noexcept { return device_; }
    [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }

  private:
    void synchronize_and_mark_idle(std::size_t slot, StreamResource& selected) noexcept
    {
      CleanupDeviceGuard guard(device_);
      check_cleanup(cudaStreamSynchronize(selected.native_handle()),
                    "cudaStreamSynchronize during stream-pool return cleanup", device_);

      std::lock_guard lock(mutex_);
      CHECK(slot < slots_.size(), slot, slots_.size());
      CHECK(slots_[slot].state == SlotState::leased || slots_[slot].state == SlotState::pending, slot);
      slots_[slot].state = SlotState::idle;
      available_.notify_one();
    }

    static void CUDART_CB return_stream_callback(void* raw_payload) noexcept
    {
      std::unique_ptr<ReturnPayload> payload(static_cast<ReturnPayload*>(raw_payload));
      payload->pool->return_stream(payload->slot);
    }

    void return_stream(std::size_t slot) noexcept { this->make_available(slot, SlotState::pending); }

    void make_available(std::size_t slot, SlotState expected_state) noexcept
    {
      detail::StreamWaiter* waiter = nullptr;
      {
        std::lock_guard lock(mutex_);
        CHECK(slot < slots_.size(), slot, slots_.size());
        CHECK(slots_[slot].state == expected_state, slot, static_cast<int>(slots_[slot].state),
              static_cast<int>(expected_state));

        if (waiter_head_ == nullptr)
        {
          slots_[slot].state = SlotState::idle;
          available_.notify_one();
          return;
        }

        waiter = waiter_head_;
        waiter_head_ = waiter->next_;
        if (waiter_head_ == nullptr) waiter_tail_ = nullptr;
        waiter->next_ = nullptr;
        waiter->queued_ = false;
        slots_[slot].state = SlotState::leased;
      }

      try
      {
        waiter->notify(Stream(std::make_shared<Stream::State>(*this, slot)));
      }
      catch (...)
      {
        PANIC("failed to deliver an available CUDA stream to a queued task", device_, slot);
      }
    }

    int device_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::vector<Slot> slots_;
    detail::StreamWaiter* waiter_head_ = nullptr;
    detail::StreamWaiter* waiter_tail_ = nullptr;
    bool stopping_ = false;
};

StreamPool::StreamPool(Config config) : impl_(std::make_unique<Impl>(config)) {}

StreamPool::~StreamPool() = default;

std::optional<Stream> StreamPool::try_acquire()
{
  auto const slot = impl_->try_acquire();
  if (!slot.has_value())
  {
    return std::nullopt;
  }
  try
  {
    return Stream(std::make_shared<Stream::State>(*impl_, *slot));
  }
  catch (...)
  {
    impl_->release_leased_without_work(*slot);
    throw;
  }
}

Stream StreamPool::acquire()
{
  std::size_t const slot = impl_->acquire();
  try
  {
    return Stream(std::make_shared<Stream::State>(*impl_, slot));
  }
  catch (...)
  {
    impl_->release_leased_without_work(slot);
    throw;
  }
}

std::optional<Stream> StreamPool::acquire_or_enqueue(detail::StreamWaiter& waiter) noexcept
{
  auto const slot = impl_->acquire_or_enqueue(waiter);
  if (!slot) return std::nullopt;

  try
  {
    return Stream(std::make_shared<Stream::State>(*impl_, *slot));
  }
  catch (...)
  {
    impl_->release_leased_without_work(*slot);
    PANIC("failed to construct a CUDA stream lease for a queued task", impl_->device(), *slot);
  }
}

int StreamPool::device() const noexcept { return impl_->device(); }

std::size_t StreamPool::size() const noexcept { return impl_->size(); }

std::size_t StreamPool::idle_stream_count() const noexcept { return impl_->count(Impl::SlotState::idle); }

std::size_t StreamPool::leased_stream_count() const noexcept { return impl_->count(Impl::SlotState::leased); }

std::size_t StreamPool::pending_stream_count() const noexcept { return impl_->count(Impl::SlotState::pending); }

void StreamPool::synchronize() { impl_->synchronize(); }

Stream::State::~State()
{
  if (pool_ != nullptr)
  {
    pool_->release_when_idle(slot_);
  }
}

int Stream::State::device() const noexcept { return pool_->stream(slot_).device(); }

cudaStream_t Stream::State::native_handle() const noexcept { return pool_->stream(slot_).native_handle(); }

void Stream::State::synchronize() const { pool_->stream(slot_).synchronize(); }

Completion Stream::State::record_completion() const
{
  StreamResource const& selected = pool_->stream(slot_);
  auto state = std::make_shared<Completion::State>(selected.device());
  state->record(selected.device(), selected.native_handle());
  return Completion(std::move(state));
}

void Stream::State::wait_on(Completion const& completion) const
{
  CHECK(completion.state_ != nullptr);
  StreamResource const& selected = pool_->stream(slot_);
  ScopedDevice guard(selected.device());
  check(cudaStreamWaitEvent(selected.native_handle(), completion.state_->native_handle()), "cudaStreamWaitEvent",
        selected.device());
}

} // namespace uni20::cuda
