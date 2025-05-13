/// \file static_vector.hpp
/// \brief Defines static_vector, a fixed-capacity, stack-allocated vector with a `std::vector`-like API.
///
/// `static_vector<T, N>` is a lightweight container that avoids heap allocation
/// by storing elements in a statically allocated buffer of capacity `N`.
/// It supports in-place construction and destruction of elements up to `N`
/// and provides the usual accessors and iterators. To get bounds checking compile with
/// DEBUG options to get precondtion checks.
/// A specialization for `MaxSize == 0` exists to allow seamless use in generic code.

#pragma once

#include "common/trace.hpp"
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

/// \brief A fixed-capacity vector with stack storage and `std::vector`-like API.
///
/// \tparam T Element type
/// \tparam MaxSize Maximum number of elements (capacity, fixed at compile time)
template <typename T, std::size_t MaxSize> class static_vector {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = T const&;
    using pointer = T*;
    using const_pointer = T const*;
    using iterator = T*;
    using const_iterator = T const*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /// \brief Default-construct an empty static_vector.
    static_vector() noexcept : size_(0) {}

    /// \brief Construct with `count` default-initialized elements.
    /// \pre `count <= MaxSize`
    explicit static_vector(size_type count) : size_(count)
    {
      DEBUG_PRECONDITION(count <= MaxSize);
      for (size_type i = 0; i < size_; ++i)
      {
        ::new (data() + i) T();
      }
    }

    /// \brief Construct with `count` copies of `value`.
    static_vector(size_type count, T const& value) : size_(count)
    {
      DEBUG_PRECONDITION(count <= MaxSize);
      for (size_type i = 0; i < size_; ++i)
      {
        ::new (data() + i) T(value);
      }
    }

    /// \brief Construct from an initializer list.
    static_vector(std::initializer_list<T> init) : size_(init.size())
    {
      DEBUG_PRECONDITION(init.size() <= MaxSize);
      size_type i = 0;
      for (T const& v : init)
      {
        ::new (data() + i++) T(v);
      }
    }

    /// \brief Copy constructor.
    static_vector(static_vector const& o) : size_(o.size_)
    {
      for (size_type i = 0; i < size_; ++i)
      {
        ::new (data() + i) T(o.data()[i]);
      }
    }

    /// \brief Move constructor.
    static_vector(static_vector&& o) noexcept(std::is_nothrow_move_constructible<T>::value) : size_(o.size_)
    {
      for (size_type i = 0; i < size_; ++i)
      {
        ::new (data() + i) T(std::move(o.data()[i]));
        o.data()[i].~T();
      }
      o.size_ = 0;
    }

    /// \brief Destructor. Destroys all elements in-place.
    ~static_vector() { clear(); }

    /// \brief Copy assignment.
    static_vector& operator=(static_vector const& o)
    {
      if (this != &o)
      {
        clear();
        size_ = o.size_;
        for (size_type i = 0; i < size_; ++i)
        {
          ::new (data() + i) T(o.data()[i]);
        }
      }
      return *this;
    }

    /// \brief Move assignment.
    static_vector& operator=(static_vector&& o) noexcept(std::is_nothrow_move_assignable<T>::value)
    {
      if (this != &o)
      {
        clear();
        size_ = o.size_;
        for (size_type i = 0; i < size_; ++i)
        {
          ::new (data() + i) T(std::move(o.data()[i]));
          o.data()[i].~T();
        }
        o.size_ = 0;
      }
      return *this;
    }

    /// \brief Assign from an initializer list.
    static_vector& operator=(std::initializer_list<T> init)
    {
      clear();
      size_ = init.size();
      DEBUG_PRECONDITION(size_ <= MaxSize);
      size_type i = 0;
      for (T const& v : init)
      {
        ::new (data() + i++) T(v);
      }
      return *this;
    }

    /// \brief Bounds-checked element access.
    reference at(size_type pos)
    {
      if (pos >= size_) throw std::out_of_range("static_vector::at");
      return data()[pos];
    }

    const_reference at(size_type pos) const
    {
      if (pos >= size_) throw std::out_of_range("static_vector::at");
      return data()[pos];
    }

    reference operator[](size_type pos)
    {
      DEBUG_PRECONDITION(pos < size_);
      return data()[pos];
    }

    const_reference operator[](size_type pos) const
    {
      DEBUG_PRECONDITION(pos < size_);
      return data()[pos];
    }

    reference front()
    {
      DEBUG_PRECONDITION(size_ > 0);
      return data()[0];
    }

    const_reference front() const
    {
      DEBUG_PRECONDITION(size_ > 0);
      return data()[0];
    }

    reference back()
    {
      DEBUG_PRECONDITION(size_ > 0);
      return data()[size_ - 1];
    }

    const_reference back() const
    {
      DEBUG_PRECONDITION(size_ > 0);
      return data()[size_ - 1];
    }

    pointer data() noexcept { return reinterpret_cast<T*>(storage_.buf); }
    const_pointer data() const noexcept { return reinterpret_cast<T const*>(storage_.buf); }

    iterator begin() noexcept { return data(); }
    const_iterator begin() const noexcept { return data(); }
    const_iterator cbegin() const noexcept { return data(); }

    iterator end() noexcept { return data() + size_; }
    const_iterator end() const noexcept { return data() + size_; }
    const_iterator cend() const noexcept { return data() + size_; }

    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    static constexpr size_type max_size() noexcept { return MaxSize; }
    static constexpr size_type capacity() noexcept { return MaxSize; }

    /// \brief Destroy all elements and reset size to zero.
    void clear() noexcept
    {
      for (size_type i = 0; i < size_; ++i)
      {
        data()[i].~T();
      }
      size_ = 0;
    }

    /// \brief Append a new copy of `value` to the end.
    void push_back(T const& value)
    {
      DEBUG_PRECONDITION(size_ < MaxSize);
      ::new (data() + size_) T(value);
      ++size_;
    }

    /// \brief Append a new moved `value` to the end.
    void push_back(T&& value)
    {
      DEBUG_PRECONDITION(size_ < MaxSize);
      ::new (data() + size_) T(std::move(value));
      ++size_;
    }

    /// \brief Construct a new element in-place at the end.
    template <class... Args> void emplace_back(Args&&... args)
    {
      DEBUG_PRECONDITION(size_ < MaxSize);
      ::new (data() + size_) T(std::forward<Args>(args)...);
      ++size_;
    }

    /// \brief Remove the last element.
    void pop_back()
    {
      DEBUG_PRECONDITION(size_ > 0);
      data()[--size_].~T();
    }

    /// \brief Resize the vector to `count` elements.
    void resize(size_type count)
    {
      DEBUG_PRECONDITION(count <= MaxSize);
      if (count < size_)
      {
        for (size_type i = count; i < size_; ++i)
        {
          data()[i].~T();
        }
      }
      else
      {
        for (size_type i = size_; i < count; ++i)
        {
          ::new (data() + i) T();
        }
      }
      size_ = count;
    }

    /// \brief Swap contents with another static_vector.
    void swap(static_vector& other) noexcept(std::is_nothrow_swappable<T>::value)
    {
      size_type m = size_ < other.size_ ? size_ : other.size_;
      for (size_type i = 0; i < m; ++i)
      {
        using std::swap;
        swap(data()[i], other.data()[i]);
      }
      if (size_ < other.size_)
      {
        for (size_type i = m; i < other.size_; ++i)
        {
          emplace_back(std::move(other.data()[i]));
        }
        other.resize(m);
      }
      else if (size_ > other.size_)
      {
        for (size_type i = m; i < size_; ++i)
        {
          other.emplace_back(std::move(data()[i]));
        }
        resize(m);
      }
    }

  private:
    struct Storage
    {
        alignas(T) unsigned char buf[MaxSize * sizeof(T)];
    } storage_;
    size_type size_;
};

/// \brief Specialization of static_vector for MaxSize == 0.
///
/// Provides full API for compatibility with generic code.
/// All modifying operations trigger a PANIC or throw on access.
template <typename T> class static_vector<T, 0> {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = T const&;
    using pointer = T*;
    using const_pointer = T const*;
    using iterator = T*;
    using const_iterator = T const*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static_vector() noexcept = default;
    explicit static_vector(size_type count) { DEBUG_CHECK_EQUAL(count, 0); }
    static_vector(size_type count, T const&) { DEBUG_CHECK_EQUAL(count, 0); }
    static_vector(std::initializer_list<T> init) { DEBUG_CHECK_EQUAL(init.size(), 0); }

    static_vector(static_vector const&) = default;
    static_vector(static_vector&&) noexcept = default;
    static_vector& operator=(static_vector const&) = default;
    static_vector& operator=(static_vector&&) noexcept = default;
    static_vector& operator=(std::initializer_list<T> init)
    {
      DEBUG_CHECK_EQUAL(init.size(), 0);
      return *this;
    }

    constexpr bool empty() const noexcept { return true; }
    constexpr size_type size() const noexcept { return 0; }
    static constexpr size_type max_size() noexcept { return 0; }
    static constexpr size_type capacity() noexcept { return 0; }

    void clear() noexcept {}
    void resize(size_type count) { DEBUG_CHECK_EQUAL(count, 0); }

    void push_back(T const&) { PANIC("unexpected: push_back() on a static_vector of size 0"); }
    void push_back(T&&) { PANIC("unexpected: push_back() on a static_vector of size 0"); }
    template <typename... Args> void emplace_back(Args&&...)
    {
      PANIC("unexpected: emplace_back() on a static_vector of size 0");
    }
    void pop_back() { PANIC("unexpected: pop_back() on a static_vector of size 0"); }

    reference at(size_type) { throw std::out_of_range("static_vector<..., 0>::at"); }
    const_reference at(size_type) const { throw std::out_of_range("static_vector<..., 0>::at"); }

    reference operator[](size_type) { PANIC("unexpected: operator[] on a static_vector of size 0"); }
    const_reference operator[](size_type) const { PANIC("unexpected: operator[] on a static_vector of size 0"); }

    reference front() { PANIC("unexpected: front() on a static_vector of size 0"); }
    const_reference front() const
    {
      PANIC("unexpected: front() on a static_vector of size 0");
      std::abort();
    }

    reference back()
    {
      PANIC("unexpected: back() on a static_vector of size 0");
      std::abort();
    }
    const_reference back() const
    {
      PANIC("unexpected: back() on a static_vector of size 0");
      std::abort();
    }

    pointer data() noexcept { return nullptr; }
    const_pointer data() const noexcept { return nullptr; }

    iterator begin() noexcept { return nullptr; }
    const_iterator begin() const noexcept { return nullptr; }
    const_iterator cbegin() const noexcept { return nullptr; }

    iterator end() noexcept { return nullptr; }
    const_iterator end() const noexcept { return nullptr; }
    const_iterator cend() const noexcept { return nullptr; }

    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

    void swap(static_vector&) noexcept {}

  private:
    // no storage
};
