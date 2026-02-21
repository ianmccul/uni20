/// \file async_node.hpp

#pragma once

#include "common/demangle.hpp"
#include "config.hpp"
#include <atomic>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace uni20::async
{

/// \brief Records node information for DAG/debugging visualization.
///
/// Each NodeInfo represents a single instance of an object participating
/// in the dependency DAG (e.g., an Async<T>). It is assigned a unique global
/// index at construction, and retains a record of the object’s address and type.
///
/// NodeInfo objects are always heap-allocated and intentionally leaked for
/// process-lifetime diagnostics. Each node is globally unique for the process lifetime.
class NodeInfo {
  public:
    /// \brief Deleted default constructor. NodeInfo can only be constructed via \ref create.
    NodeInfo() = delete;

    /// \brief Returns the raw address of the referenced value.
    void const* address() const { return address_; }

    /// \brief Returns a human-readable, demangled type name.
    ///
    /// The string is obtained from an interned map keyed by the mangled type name.
    /// The returned string_view is valid for the process lifetime.
    ///
    /// \note Thread-safe.
    std::string_view type() const
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      auto [it, inserted] = map_.try_emplace(type_key_, uni20::demangle::demangle(type_key_));
      return std::string_view(it->second);
    }

    /// \brief Returns the unique global index of this node.
    ///
    /// \return Monotonically increasing value, unique across all NodeInfo instances for this process.
    uint64_t global_index() const { return global_index_; }

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

  private:
    template <typename T>
    NodeInfo(T const* value)
        : address_(static_cast<void const*>(value)), type_key_(typeid(T).name()), global_index_(next_global_++)
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
    void const* address_;   ///< Address of the referenced value.
    char const* type_key_;  ///< Mangled type name (from typeid).
    uint64_t global_index_; ///< Globally unique node index.
#if 0 && UNI20_HAS_STACKTRACE
    std::stacktrace stack_; ///< stacktrace of where the node was constructed
#endif
};

} // namespace uni20::async
