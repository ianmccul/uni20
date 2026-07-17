#include <uni20/backend/cuda/buffer.hpp>

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/common/trace.hpp>

#include <algorithm>
#include <utility>

namespace uni20::cuda
{

namespace
{

void check_cleanup(cudaError_t status, char const* operation, int device) noexcept
{
  if (status == cudaSuccess)
  {
    return;
  }
  PANIC("CUDA cleanup operation failed", operation, device, cudaGetErrorName(status), cudaGetErrorString(status));
}

class CleanupDeviceGuard {
  public:
    explicit CleanupDeviceGuard(int device) noexcept
    {
      check_cleanup(cudaGetDevice(&previous_device_), "cudaGetDevice", device);
      restore_ = previous_device_ != device;
      if (restore_)
      {
        check_cleanup(cudaSetDevice(device), "cudaSetDevice", device);
      }
    }

    ~CleanupDeviceGuard()
    {
      if (restore_)
      {
        check_cleanup(cudaSetDevice(previous_device_), "cudaSetDevice restore", previous_device_);
      }
    }

  private:
    int previous_device_ = -1;
    bool restore_ = false;
};

void synchronize_after_failed_publication(Stream const& stream, char const* operation) noexcept
{
  try
  {
    stream.synchronize();
  }
  catch (...)
  {
    PANIC("CUDA buffer completion publication failed and stream synchronization also failed", operation,
          stream.device());
  }
}

} // namespace

DeviceContext::DeviceContext(Config config)
    : device_(config.device),
      streams_(
          {.device = config.device.ordinal(), .stream_count = config.stream_count, .stream_flags = config.stream_flags})
{}

BufferStorage::BufferStorage(DeviceContext& context, std::size_t size_bytes)
    : context_(&context), size_bytes_(size_bytes)
{
  if (size_bytes_ == 0)
  {
    return;
  }

  ScopedDevice guard(context.device().ordinal());
  check(cudaMalloc(&data_, size_bytes_), "cudaMalloc", context.device().ordinal());
}

BufferStorage::BufferStorage(BufferStorage&& other) noexcept
    : context_(other.context_), data_(other.data_), size_bytes_(other.size_bytes_),
      writer_completion_(std::move(other.writer_completion_)), reader_completions_(std::move(other.reader_completions_))
{
  other.context_ = nullptr;
  other.data_ = nullptr;
  other.size_bytes_ = 0;
}

BufferStorage& BufferStorage::operator=(BufferStorage&& other) noexcept
{
  if (this != &other)
  {
    this->reset();
    context_ = other.context_;
    data_ = other.data_;
    size_bytes_ = other.size_bytes_;
    writer_completion_ = std::move(other.writer_completion_);
    reader_completions_ = std::move(other.reader_completions_);
    other.context_ = nullptr;
    other.data_ = nullptr;
    other.size_bytes_ = 0;
  }
  return *this;
}

BufferStorage::~BufferStorage() { this->reset(); }

DeviceContext& BufferStorage::context() const
{
  CHECK(context_ != nullptr);
  return *context_;
}

Device BufferStorage::device() const { return this->context().device(); }

void BufferStorage::install_read_waits(Stream const& stream) const
{
  CHECK_EQUAL(stream.device(), this->device().ordinal());

  Completion writer_completion;
  {
    std::lock_guard lock(this->context().state_mutex_);
    writer_completion = writer_completion_;
  }

  if (writer_completion)
  {
    stream.wait_on(writer_completion);
  }
}

void BufferStorage::install_write_waits(Stream const& stream) const
{
  CHECK_EQUAL(stream.device(), this->device().ordinal());

  Completion writer_completion;
  std::vector<Completion> reader_completions;
  {
    std::lock_guard lock(this->context().state_mutex_);
    writer_completion = writer_completion_;
    std::erase_if(reader_completions_, [](Completion const& completion) { return completion.ready(); });
    reader_completions = reader_completions_;
  }

  if (writer_completion)
  {
    stream.wait_on(writer_completion);
  }
  for (Completion const& reader_completion : reader_completions)
  {
    stream.wait_on(reader_completion);
  }
}

void BufferStorage::publish_read_after(Stream const& stream) const noexcept
{
  try
  {
    this->publish_reader(stream.record_completion());
  }
  catch (...)
  {
    synchronize_after_failed_publication(stream, "publish CUDA buffer reader completion");
  }
}

void BufferStorage::publish_write_after(Stream const& stream) const noexcept
{
  try
  {
    this->publish_writer(stream.record_completion());
  }
  catch (...)
  {
    synchronize_after_failed_publication(stream, "publish CUDA buffer writer completion");
  }
}

void BufferStorage::publish_reader(Completion const& completion) const
{
  std::lock_guard lock(this->context().state_mutex_);
  if (reader_completions_.size() == reader_completions_.capacity())
  {
    std::erase_if(reader_completions_, [](Completion const& reader_completion) { return reader_completion.ready(); });
  }
  reader_completions_.push_back(completion);
}

void BufferStorage::publish_writer(Completion const& completion) const
{
  std::lock_guard lock(this->context().state_mutex_);
  writer_completion_ = completion;
  reader_completions_.clear();
}

void BufferStorage::synchronize() const
{
  Completion writer_completion;
  std::vector<Completion> reader_completions;
  {
    std::lock_guard lock(this->context().state_mutex_);
    writer_completion = writer_completion_;
    reader_completions = reader_completions_;
  }
  if (writer_completion)
  {
    writer_completion.synchronize();
  }
  for (Completion const& reader_completion : reader_completions)
  {
    reader_completion.synchronize();
  }
}

void BufferStorage::reset() noexcept
{
  if (context_ == nullptr)
  {
    return;
  }

  Completion writer_completion;
  std::vector<Completion> reader_completions;
  {
    std::lock_guard lock(context_->state_mutex_);
    writer_completion = std::move(writer_completion_);
    reader_completions = std::move(reader_completions_);
  }
  try
  {
    if (writer_completion)
    {
      writer_completion.synchronize();
    }
    for (Completion const& reader_completion : reader_completions)
    {
      reader_completion.synchronize();
    }
  }
  catch (...)
  {
    PANIC("CUDA buffer completion synchronization failed during cleanup", context_->device().ordinal());
  }

  if (data_ != nullptr)
  {
    int const device = context_->device().ordinal();
    CleanupDeviceGuard guard(device);
    check_cleanup(cudaFree(data_), "cudaFree", device);
  }

  context_ = nullptr;
  data_ = nullptr;
  size_bytes_ = 0;
}

} // namespace uni20::cuda
