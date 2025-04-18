#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

// A stack-allocated, fixed-capacity vector with API similar to std::vector.
// No heap allocation up to MaxSize elements.
template <typename T, std::size_t MaxSize> class static_vector {
  public:
    // Member types
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

    // Constructors
    static_vector() noexcept : size_(0) {}

    explicit static_vector(size_type count) : size_(count)
    {
      assert(count <= MaxSize);
      for (size_type i = 0; i < size_; ++i)
      {
        ::new (data() + i) T();
      }
    }

    static_vector(size_type count, T const& value) : size_(count)
    {
      assert(count <= MaxSize);
      for (size_type i = 0; i < size_; ++i)
      {
        ::new (data() + i) T(value);
      }
    }

    static_vector(std::initializer_list<T> init) : size_(init.size())
    {
      assert(init.size() <= MaxSize);
      size_type i = 0;
      for (T const& v : init)
      {
        ::new (data() + i++) T(v);
      }
    }

    // Copy
    static_vector(static_vector const& o) : size_(o.size_)
    {
      for (size_type i = 0; i < size_; ++i)
      {
        ::new (data() + i) T(o.data()[i]);
      }
    }

    // Move
    static_vector(static_vector&& o) noexcept(std::is_nothrow_move_constructible<T>::value) : size_(o.size_)
    {
      for (size_type i = 0; i < size_; ++i)
      {
        ::new (data() + i) T(std::move(o.data()[i]));
        o.data()[i].~T();
      }
      o.size_ = 0;
    }

    ~static_vector() { clear(); }

    // Copy assignment
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

    // Move assignment
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

    // Initializer-list assignment
    static_vector& operator=(std::initializer_list<T> init)
    {
      clear();
      size_ = init.size();
      assert(size_ <= MaxSize);
      size_type i = 0;
      for (T const& v : init)
      {
        ::new (data() + i++) T(v);
      }
      return *this;
    }

    // Element access
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
      assert(pos < size_);
      return data()[pos];
    }
    const_reference operator[](size_type pos) const
    {
      assert(pos < size_);
      return data()[pos];
    }

    reference front()
    {
      assert(size_ > 0);
      return data()[0];
    }
    const_reference front() const
    {
      assert(size_ > 0);
      return data()[0];
    }

    reference back()
    {
      assert(size_ > 0);
      return data()[size_ - 1];
    }
    const_reference back() const
    {
      assert(size_ > 0);
      return data()[size_ - 1];
    }

    pointer data() noexcept { return reinterpret_cast<T*>(storage_.buf); }
    const_pointer data() const noexcept { return reinterpret_cast<T const*>(storage_.buf); }

    // Iterators
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

    // Capacity
    bool empty() const noexcept { return size_ == 0; }
    size_type size() const noexcept { return size_; }
    static constexpr size_type max_size() noexcept { return MaxSize; }
    static constexpr size_type capacity() noexcept { return MaxSize; }

    // Modifiers
    void clear() noexcept
    {
      for (size_type i = 0; i < size_; ++i)
      {
        data()[i].~T();
      }
      size_ = 0;
    }

    void push_back(T const& value)
    {
      assert(size_ < MaxSize);
      ::new (data() + size_) T(value);
      ++size_;
    }

    void push_back(T&& value)
    {
      assert(size_ < MaxSize);
      ::new (data() + size_) T(std::move(value));
      ++size_;
    }

    template <class... Args> void emplace_back(Args&&... args)
    {
      assert(size_ < MaxSize);
      ::new (data() + size_) T(std::forward<Args>(args)...);
      ++size_;
    }

    void pop_back()
    {
      assert(size_ > 0);
      data()[--size_].~T();
    }

    void resize(size_type count)
    {
      assert(count <= MaxSize);
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
