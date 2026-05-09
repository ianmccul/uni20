/// \file async_node.hpp
/// \brief Debug metadata nodes used for async DAG visualization.

#pragma once

#include "shared_storage.hpp"
#include <uni20/common/demangle.hpp>
#include <uni20/config.hpp>
#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

namespace uni20::async
{

/// \brief Construction state reported for a debug DAG value node.
enum class NodeValueState
{
  Invalid,       ///< No storage or fallback value is available.
  Unconstructed, ///< Storage exists but does not currently contain a value.
  Constructed,  ///< A value is currently constructed.
};

namespace detail
{

/// \brief Poison-pill declaration used to detect user ADL debug-value hooks.
void uni20_async_debug_value();

/// \brief Detects a one-line debug-value overload found by argument-dependent lookup.
/// \tparam T Value type to inspect.
template <typename T>
concept HasAdlAsyncDebugValue = requires(T const& value) { std::string{uni20_async_debug_value(value)}; };

/// \brief Format extents as a compact shape summary.
/// \tparam Extents Mdspan-style extents type.
/// \param extents Shape descriptor to format.
/// \return `shape=(...)` with one entry per axis.
template <typename Extents> std::string format_shape_debug_info(Extents const& extents)
{
  using extents_type = std::remove_cvref_t<Extents>;
  std::string out = "shape=(";
  for (std::size_t axis = 0; axis < static_cast<std::size_t>(extents_type::rank()); ++axis)
  {
    if (axis != 0) out += ", ";
    out += std::to_string(static_cast<unsigned long long>(extents.extent(axis)));
  }
  out += ")";
  return out;
}

/// \brief Detects objects that expose `mdspan().extents()`.
/// \tparam T Value type to inspect.
template <typename T>
concept HasMdspanExtents = requires(T const& value, std::size_t axis) {
  std::remove_cvref_t<decltype(value.mdspan().extents())>::rank();
  value.mdspan().extents().extent(axis);
};

/// \brief Detects objects that expose `extents()`.
/// \tparam T Value type to inspect.
template <typename T>
concept HasExtents = requires(T const& value, std::size_t axis) {
  std::remove_cvref_t<decltype(value.extents())>::rank();
  value.extents().extent(axis);
};

/// \brief Normalize debug value text to a single line for Graphviz node labels.
/// \param value Debug-value string to sanitize.
/// \return Sanitized debug-value string.
inline std::string sanitize_debug_value(std::string value)
{
  for (char& ch : value)
  {
    if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
  }
  return value;
}

/// \brief Format built-in scalar values for async DAG labels.
/// \tparam T Scalar value type.
/// \param value Value to format.
/// \return One-line scalar representation, or `constructed` for non-scalars.
template <typename T> std::string scalar_debug_info(T const& value)
{
  using value_type = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<value_type, bool>)
  {
    return value ? "true" : "false";
  }
  else if constexpr (std::is_same_v<value_type, char>)
  {
    return std::string{"'"} + value + "'";
  }
  else if constexpr (std::is_integral_v<value_type>)
  {
    if constexpr (std::is_signed_v<value_type>)
      return std::to_string(static_cast<long long>(value));
    else
      return std::to_string(static_cast<unsigned long long>(value));
  }
  else if constexpr (std::is_floating_point_v<value_type>)
  {
    std::ostringstream out;
    out.precision(std::numeric_limits<value_type>::digits10);
    out << value;
    return out.str();
  }
  else if constexpr (std::is_enum_v<value_type>)
  {
    using underlying_type = std::underlying_type_t<value_type>;
    return scalar_debug_info(static_cast<underlying_type>(value));
  }
  else
  {
    return {};
  }
}

/// \brief Default one-line debug value for values without a custom ADL hook.
/// \tparam T Value type to inspect.
/// \param value Value to summarize.
/// \return Scalar value, shape summary, or empty string when no useful value is available.
template <typename T> std::string default_debug_value(T const& value)
{
  if constexpr (HasMdspanExtents<T>)
  {
    auto span = value.mdspan();
    return format_shape_debug_info(span.extents());
  }
  else if constexpr (HasExtents<T>)
  {
    return format_shape_debug_info(value.extents());
  }
  else
  {
    return scalar_debug_info(value);
  }
}

/// \brief Resolve custom or default one-line debug value for a constructed value.
/// \tparam T Value type to inspect.
/// \param value Constructed value to summarize.
/// \return One-line debug value, or empty string when no useful value is available.
template <typename T> std::string debug_value_for_value(T const& value)
{
  if constexpr (HasAdlAsyncDebugValue<T>)
  {
    return sanitize_debug_value(std::string{uni20_async_debug_value(value)});
  }
  else
  {
    return sanitize_debug_value(default_debug_value(value));
  }
}

} // namespace detail

/// \brief Records node information for DAG/debugging visualization.
///
/// Each NodeInfo represents a single instance of an object participating
/// in the dependency DAG (e.g., an Async<T>). It is assigned a unique global
/// index at construction, and retains diagnostic addresses for the async storage
/// and the constructed value when available.
///
/// NodeInfo objects are always heap-allocated and intentionally leaked for
/// process-lifetime diagnostics. Each node is globally unique for the process lifetime.
class NodeInfo {
  public:
    /// \brief Deleted default constructor. NodeInfo can only be constructed via \ref create.
    NodeInfo() = delete;

    /// \brief Returns the raw address of the current constructed value when observable.
    /// \return Constructed value address, or `nullptr` if the value is currently unconstructed.
    [[nodiscard]] void const* address() const
    {
      return value_address_observer_ ? value_address_observer_(storage_address_) : fallback_value_address_;
    }

    /// \brief Returns the raw address of the shared storage control block.
    /// \return Control-block address, or `nullptr` when the node was not storage-backed.
    [[nodiscard]] void const* storage_address() const { return storage_address_; }

    /// \brief Reports the current construction state when observable.
    /// \return Current value state for snapshot rendering.
    [[nodiscard]] NodeValueState value_state() const noexcept
    {
      if (value_constructed_observer_)
      {
        if (!storage_address_) return NodeValueState::Invalid;
        return value_constructed_observer_(storage_address_) ? NodeValueState::Constructed
                                                             : NodeValueState::Unconstructed;
      }
      return fallback_value_address_ ? NodeValueState::Constructed : NodeValueState::Invalid;
    }

    /// \brief Returns a stable label for the current construction state.
    /// \return State label suitable for Graphviz output.
    [[nodiscard]] std::string_view value_state_label() const noexcept
    {
      switch (this->value_state())
      {
      case NodeValueState::Invalid:
        return "invalid";
      case NodeValueState::Unconstructed:
        return "unconstructed";
      case NodeValueState::Constructed:
        return "constructed";
      }
      return "invalid";
    }

    /// \brief Reports whether the value is currently constructed when observable.
    /// \return `true` when \ref address is non-null.
    [[nodiscard]] bool value_constructed() const noexcept { return this->value_state() == NodeValueState::Constructed; }

    /// \brief Returns type-specific display text for the current value.
    /// \return One-line value text, such as a scalar value or `shape=(...)`, when available.
    [[nodiscard]] std::string value_text() const
    {
      auto const state = this->value_state();
      if (state != NodeValueState::Constructed) return {};
      if (!value_text_observer_) return {};
      auto const* context = value_constructed_observer_ ? storage_address_ : fallback_value_address_;
      return value_text_observer_(context);
    }

    /// \brief Returns a human-readable, demangled type name.
    ///
    /// The string is obtained from an interned map keyed by the mangled type name.
    /// The returned string_view is valid for the process lifetime.
    ///
    /// \note Thread-safe.
    [[nodiscard]] std::string_view type() const
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      auto [it, inserted] = map_.try_emplace(type_key_, uni20::demangle::demangle(type_key_));
      return std::string_view(it->second);
    }

    /// \brief Returns the unique global index of this node.
    ///
    /// \return Monotonically increasing value, unique across all NodeInfo instances for this process.
    [[nodiscard]] uint64_t global_index() const { return global_index_; }

    /// \brief Returns the stacktrace of where the node was created
#if 0 && UNI20_HAS_STACKTRACE
    std::stacktrace const& stack() const { return stack_; }
#endif

    /// \brief Factory method to create a NodeInfo for an object pointer.
    ///
    /// \tparam T  The type of the object.
    /// \param value  Pointer to the value for which a NodeInfo should be created.
    /// \return Pointer to the new NodeInfo (never deallocated).
    ///
    /// \post The returned NodeInfo outlives all references (process lifetime).
    template <typename T> static NodeInfo const* create(T const* value) { return new NodeInfo(value); }

    /// \brief Factory method to create a NodeInfo for shared async storage.
    ///
    /// \tparam T The async storage value type.
    /// \param storage Storage handle represented by this node.
    /// \return Pointer to the new NodeInfo (never deallocated).
    ///
    /// \post The returned NodeInfo outlives all references (process lifetime).
    template <typename T> static NodeInfo const* create(shared_storage<T> const& storage) { return new NodeInfo(storage); }

  private:
    using ValueAddressObserver = void const* (*)(void const*) noexcept;
    using ValueConstructedObserver = bool (*)(void const*) noexcept;
    using ValueTextObserver = std::string (*)(void const*);

    template <typename T> static std::string text_for_raw_value(void const* value)
    {
      return value ? detail::debug_value_for_value(*static_cast<T const*>(value)) : std::string{};
    }

    template <typename T> static std::string text_for_storage_value(void const* control)
    {
      auto const* value = static_cast<T const*>(shared_storage<T>::diagnostic_value_address_from_control(control));
      return value ? detail::debug_value_for_value(*value) : std::string{};
    }

    template <typename T>
    NodeInfo(T const* value)
        : fallback_value_address_(static_cast<void const*>(value)), type_key_(typeid(T).name()),
          value_text_observer_(&NodeInfo::text_for_raw_value<T>), global_index_(next_global_++)
#if 0 && UNI20_HAS_STACKTRACE
          ,
          stack_(std::stacktrace::current())
#endif
    {}

    template <typename T>
    explicit NodeInfo(shared_storage<T> const& storage)
        : storage_address_(storage.control_address()), fallback_value_address_(static_cast<void const*>(storage.get())),
          type_key_(typeid(T).name()), value_address_observer_(&shared_storage<T>::diagnostic_value_address_from_control),
          value_constructed_observer_(&shared_storage<T>::diagnostic_constructed_from_control),
          value_text_observer_(&NodeInfo::text_for_storage_value<T>),
          global_index_(next_global_++)
#if 0 && UNI20_HAS_STACKTRACE
          ,
          stack_(std::stacktrace::current())
#endif
    {}

    // global data for the map of mangled name to demangled name (protected by map_mutex_), and the global index
    inline static std::unordered_map<const char*, std::string> map_;
    inline static std::mutex map_mutex_;
    inline static std::atomic<uint64_t> next_global_ = 0;

    // Per-instance data:
    void const* storage_address_{nullptr};                    ///< Address of the async storage control block.
    void const* fallback_value_address_{nullptr};             ///< Value address used when no observer is available.
    char const* type_key_;                                    ///< Mangled type name (from typeid).
    ValueAddressObserver value_address_observer_{};           ///< Observer for current value address.
    ValueConstructedObserver value_constructed_observer_{};   ///< Observer for current construction state.
    ValueTextObserver value_text_observer_{};                 ///< Observer for one-line value text.
    uint64_t global_index_;                                   ///< Globally unique node index.
#if 0 && UNI20_HAS_STACKTRACE
    std::stacktrace stack_; ///< stacktrace of where the node was constructed
#endif
};

} // namespace uni20::async
