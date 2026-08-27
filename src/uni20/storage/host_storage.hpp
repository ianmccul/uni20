#pragma once

#include <uni20/common/aligned_buffer.hpp>
#include <uni20/config.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/mdspan/mdspan.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace uni20
{

/// \brief Owning contiguous host buffer used by `HostStorage`.
/// \details Allocation and growth leave `uninitialized_ok` elements unspecified.
///          Other element types are default-constructed and destroyed normally.
///          Copying preserves every stored object representation, and resizing
///          preserves the common prefix just like `std::vector::resize`.
/// \tparam ElementType Element type held in pageable host memory.
template <typename ElementType> class HostBuffer {
  public:
    using value_type = ElementType;
    using size_type = std::size_t;
    using iterator = value_type*;
    using const_iterator = value_type const*;

    static constexpr bool initializes_elements = !uninitialized_ok<value_type>;

    HostBuffer() noexcept = default;

    /// \brief Allocate storage for `size` elements.
    /// \details Values are unspecified when `value_type` satisfies
    ///          `uninitialized_ok`; otherwise elements are default-constructed.
    explicit HostBuffer(size_type size) : data_(allocate(size)), size_(size) {}

    /// \brief Allocate and fill storage with one value.
    HostBuffer(size_type size, value_type const& value) : HostBuffer(size) { std::fill_n(data_, size_, value); }

    HostBuffer(HostBuffer const& other) : HostBuffer(other.size_) { this->copy_from(other.data_, other.size_); }

    HostBuffer(HostBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0))
    {}

    ~HostBuffer() { release(data_, size_); }

    auto operator=(HostBuffer const& other) -> HostBuffer&
    {
      if (this == &other) return *this;
      HostBuffer replacement(other);
      this->swap(replacement);
      return *this;
    }

    auto operator=(HostBuffer&& other) noexcept -> HostBuffer&
    {
      if (this == &other) return *this;
      release(data_, size_);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      return *this;
    }

    /// \brief Replace the allocation while preserving the common prefix.
    /// \details Newly added `uninitialized_ok` elements have unspecified values.
    void resize(size_type size)
    {
      if (size == size_) return;

      value_type* replacement = allocate(size);
      size_type const copied_size = std::min(size_, size);
      try
      {
        copy_values(replacement, data_, copied_size);
      }
      catch (...)
      {
        release(replacement, size);
        throw;
      }

      release(data_, size_);
      data_ = replacement;
      size_ = size;
    }

    void clear() noexcept
    {
      release(data_, size_);
      data_ = nullptr;
      size_ = 0;
    }

    void swap(HostBuffer& other) noexcept
    {
      using std::swap;
      swap(data_, other.data_);
      swap(size_, other.size_);
    }

    [[nodiscard]] auto data() noexcept -> value_type* { return data_; }
    [[nodiscard]] auto data() const noexcept -> value_type const* { return data_; }
    [[nodiscard]] auto size() const noexcept -> size_type { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] auto begin() noexcept -> iterator { return data_; }
    [[nodiscard]] auto begin() const noexcept -> const_iterator { return data_; }
    [[nodiscard]] auto end() noexcept -> iterator { return size_ == 0 ? data_ : data_ + size_; }
    [[nodiscard]] auto end() const noexcept -> const_iterator { return size_ == 0 ? data_ : data_ + size_; }

    [[nodiscard]] auto operator[](size_type index) noexcept -> value_type& { return data_[index]; }
    [[nodiscard]] auto operator[](size_type index) const noexcept -> value_type const& { return data_[index]; }

    friend bool operator==(HostBuffer const& lhs, HostBuffer const& rhs)
    {
      return lhs.size_ == rhs.size_ && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

  private:
    static auto allocate(size_type size) -> value_type*
    {
      if (size == 0) return nullptr;
      if (size > std::numeric_limits<size_type>::max() / sizeof(value_type)) throw std::bad_array_new_length{};

      constexpr size_type alignment = std::max<size_type>(64, alignof(value_type));
      auto* result = static_cast<value_type*>(detail::allocate_raw(sizeof(value_type) * size, alignment));
      if constexpr (!uninitialized_ok<value_type>)
      {
        try
        {
          std::uninitialized_default_construct_n(result, size);
        }
        catch (...)
        {
          detail::aligned_deleter<value_type>{}(result);
          throw;
        }
      }
      return result;
    }

    static void release(value_type* data, size_type size) noexcept
    {
      if (data == nullptr) return;
      if constexpr (!uninitialized_ok<value_type>)
      {
        std::destroy_n(data, size);
      }
      detail::aligned_deleter<value_type>{}(data);
    }

    static void copy_values(value_type* output, value_type const* input, size_type size)
    {
      if (size == 0) return;
      if constexpr (uninitialized_ok<value_type>)
      {
        std::memcpy(output, input, size * sizeof(value_type));
      }
      else
      {
        std::copy_n(input, size, output);
      }
    }

    void copy_from(value_type const* input, size_type size) { copy_values(data_, input, size); }

    value_type* data_ = nullptr;
    size_type size_ = 0;
};

template <typename ElementType> void swap(HostBuffer<ElementType>& lhs, HostBuffer<ElementType>& rhs) noexcept
{
  lhs.swap(rhs);
}

/// \brief Default pageable-host tensor storage policy.
/// \details Uses `HostBuffer` for aligned allocation and selects the available
///          host BLAS/LAPACK and reference backends.
struct HostStorage
{
    /// \brief Guaranteed byte alignment of every `HostBuffer` allocation.
    static constexpr std::size_t allocation_alignment = 64;

    template <typename ElementType> using storage_t = HostBuffer<ElementType>;

    template <typename ElementType> static auto make_handle(storage_t<ElementType>& storage) noexcept -> ElementType*
    {
      return storage.data();
    }

    template <typename ElementType>
    static auto make_handle(storage_t<ElementType> const& storage) noexcept -> ElementType const*
    {
      return storage.data();
    }

#if UNI20_BACKEND_BLAS
    using backend_selector_type =
        linalg::backend_list<linalg::LapackBackend, linalg::BlasBackend, linalg::CpuReferenceBackend>;
#else
    using backend_selector_type = linalg::backend_list<linalg::LapackBackend, linalg::CpuReferenceBackend>;
#endif

    /// \brief Return the default ordered linalg backends for pageable host storage.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
#if UNI20_BACKEND_BLAS
      return backend_selector_type{linalg::LapackBackend{}, linalg::BlasBackend{}, linalg::CpuReferenceBackend{}};
#else
      return backend_selector_type{linalg::LapackBackend{}, linalg::CpuReferenceBackend{}};
#endif
    }
};

} // namespace uni20
