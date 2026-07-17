#include <uni20/backend/cuda/runtime.hpp>

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/common/trace.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <mutex>
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

Stream::Stream(int device, unsigned int flags) : device_(device)
{
  ScopedDevice guard(device_);
  check(cudaStreamCreateWithFlags(&stream_, flags), "cudaStreamCreateWithFlags", device_);
}

Stream::Stream(Stream&& other) noexcept : device_(other.device_), stream_(other.stream_)
{
  other.device_ = -1;
  other.stream_ = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept
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

Stream::~Stream() { this->reset(); }

void Stream::synchronize() const
{
  CHECK(stream_ != nullptr);
  ScopedDevice guard(device_);
  check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", device_);
}

void Stream::reset() noexcept
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

Event::Event(int device) : device_(device)
{
  ScopedDevice guard(device_);
  check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "cudaEventCreateWithFlags", device_);
}

Event::Event(Event&& other) noexcept : device_(other.device_), event_(other.event_)
{
  other.device_ = -1;
  other.event_ = nullptr;
}

Event& Event::operator=(Event&& other) noexcept
{
  if (this != &other)
  {
    this->reset();
    device_ = other.device_;
    event_ = other.event_;
    other.device_ = -1;
    other.event_ = nullptr;
  }
  return *this;
}

Event::~Event() { this->reset(); }

void Event::record(Stream const& stream) const
{
  CHECK(event_ != nullptr);
  CHECK_EQUAL(device_, stream.device());
  ScopedDevice guard(stream.device());
  check(cudaEventRecord(event_, stream.native_handle()), "cudaEventRecord", stream.device());
}

void Event::wait_on(Stream const& stream) const
{
  CHECK(event_ != nullptr);
  ScopedDevice guard(stream.device());
  check(cudaStreamWaitEvent(stream.native_handle(), event_), "cudaStreamWaitEvent", stream.device());
}

bool Event::ready() const
{
  CHECK(event_ != nullptr);
  ScopedDevice guard(device_);
  cudaError_t const status = cudaEventQuery(event_);
  if (status == cudaErrorNotReady)
  {
    return false;
  }
  check(status, "cudaEventQuery", device_);
  return true;
}

void Event::synchronize() const
{
  CHECK(event_ != nullptr);
  ScopedDevice guard(device_);
  check(cudaEventSynchronize(event_), "cudaEventSynchronize", device_);
}

void Event::reset() noexcept
{
  if (event_ == nullptr)
  {
    return;
  }
  CleanupDeviceGuard guard(device_);
  check_cleanup(cudaEventDestroy(event_), "cudaEventDestroy", device_);
  device_ = -1;
  event_ = nullptr;
}

Completion::Completion(Event event) : event_(std::make_shared<Event>(std::move(event))) {}

int Completion::device() const noexcept { return event_ == nullptr ? -1 : event_->device(); }

cudaEvent_t Completion::native_handle() const noexcept { return event_ == nullptr ? nullptr : event_->native_handle(); }

void Completion::wait_on(Stream const& stream) const
{
  CHECK(event_ != nullptr);
  event_->wait_on(stream);
}

bool Completion::ready() const
{
  CHECK(event_ != nullptr);
  return event_->ready();
}

void Completion::synchronize() const
{
  CHECK(event_ != nullptr);
  event_->synchronize();
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

        Stream stream;
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

    [[nodiscard]] Stream& stream(std::size_t slot)
    {
      CHECK(slot < slots_.size(), slot, slots_.size());
      return slots_[slot].stream;
    }

    [[nodiscard]] Completion submit(std::size_t slot)
    {
      Stream& selected = this->stream(slot);
      try
      {
        Event event(device_);
        event.record(selected);
        Completion completion(std::move(event));
        auto* payload = new ReturnPayload{this, slot};

        {
          std::lock_guard lock(mutex_);
          CHECK(slots_[slot].state == SlotState::leased, slot);
          slots_[slot].state = SlotState::pending;
        }

        // The event publishes data readiness. The following host function is a
        // separate admission-control boundary that makes the stream leasable.
        ScopedDevice guard(device_);
        cudaError_t const status = cudaLaunchHostFunc(selected.native_handle(), &Impl::return_stream_callback, payload);
        if (status != cudaSuccess)
        {
          delete payload;
        }
        check(status, "cudaLaunchHostFunc", device_);

        return completion;
      }
      catch (...)
      {
        // Submission setup can fail after the caller has queued work. Complete
        // that work before making the slot available and propagating the error.
        CleanupDeviceGuard guard(device_);
        check_cleanup(cudaStreamSynchronize(selected.native_handle()),
                      "cudaStreamSynchronize after failed stream submission", device_);
        std::lock_guard lock(mutex_);
        CHECK(slots_[slot].state == SlotState::leased || slots_[slot].state == SlotState::pending, slot);
        slots_[slot].state = SlotState::idle;
        throw;
      }
    }

    void release_without_submission(std::size_t slot) noexcept
    {
      std::lock_guard lock(mutex_);
      CHECK(slot < slots_.size(), slot, slots_.size());
      CHECK(slots_[slot].state == SlotState::leased, slot);
      slots_[slot].state = SlotState::idle;
    }

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
    static void CUDART_CB return_stream_callback(void* raw_payload) noexcept
    {
      std::unique_ptr<ReturnPayload> payload(static_cast<ReturnPayload*>(raw_payload));
      payload->pool->return_stream(payload->slot);
    }

    void return_stream(std::size_t slot) noexcept
    {
      std::lock_guard lock(mutex_);
      CHECK(slot < slots_.size(), slot, slots_.size());
      CHECK(slots_[slot].state == SlotState::pending, slot);
      slots_[slot].state = SlotState::idle;
    }

    int device_;
    mutable std::mutex mutex_;
    std::vector<Slot> slots_;
    bool stopping_ = false;
};

StreamPool::StreamPool(Config config) : impl_(std::make_unique<Impl>(config)) {}

StreamPool::~StreamPool() = default;

std::optional<StreamPool::Lease> StreamPool::try_acquire()
{
  auto const slot = impl_->try_acquire();
  if (!slot.has_value())
  {
    return std::nullopt;
  }
  return Lease(*this, *slot);
}

int StreamPool::device() const noexcept { return impl_->device(); }

std::size_t StreamPool::size() const noexcept { return impl_->size(); }

std::size_t StreamPool::idle_stream_count() const noexcept { return impl_->count(Impl::SlotState::idle); }

std::size_t StreamPool::leased_stream_count() const noexcept { return impl_->count(Impl::SlotState::leased); }

std::size_t StreamPool::pending_stream_count() const noexcept { return impl_->count(Impl::SlotState::pending); }

void StreamPool::synchronize() { impl_->synchronize(); }

Completion StreamPool::submit(std::size_t slot) { return impl_->submit(slot); }

void StreamPool::release_without_submission(std::size_t slot) noexcept { impl_->release_without_submission(slot); }

StreamPool::Lease::Lease(Lease&& other) noexcept : pool_(other.pool_), slot_(other.slot_)
{
  other.pool_ = nullptr;
  other.slot_ = 0;
}

StreamPool::Lease& StreamPool::Lease::operator=(Lease&& other) noexcept
{
  if (this != &other)
  {
    CHECK(pool_ == nullptr, "overwriting an active CUDA stream lease");
    pool_ = other.pool_;
    slot_ = other.slot_;
    other.pool_ = nullptr;
    other.slot_ = 0;
  }
  return *this;
}

StreamPool::Lease::~Lease() { CHECK(pool_ == nullptr, "CUDA stream lease was neither submitted nor released"); }

Stream& StreamPool::Lease::stream() const noexcept
{
  CHECK(pool_ != nullptr);
  return pool_->impl_->stream(slot_);
}

Completion StreamPool::Lease::submit()
{
  CHECK(pool_ != nullptr);
  try
  {
    Completion completion = pool_->submit(slot_);
    pool_ = nullptr;
    slot_ = 0;
    return completion;
  }
  catch (...)
  {
    pool_ = nullptr;
    slot_ = 0;
    throw;
  }
}

void StreamPool::Lease::release_without_submission() noexcept
{
  CHECK(pool_ != nullptr);
  pool_->release_without_submission(slot_);
  pool_ = nullptr;
  slot_ = 0;
}

} // namespace uni20::cuda
