/// \file buffers.hpp
/// \brief Awaitable gates for Async<T>: snapshot‐reads and in‐place writes.
/// \ingroup async_api

#pragma once

#include "async_task.hpp"
#include "common/trace.hpp"
#include "epoch_context.hpp"

#include <coroutine>
#include <utility>

namespace uni20::async
{

template <typename T> class Async;

/// \brief RAII handle for reading the value from an Async container at a given epoch.
///
/// A ReadBuffer<T> represents a read-only access to the value of an Async<T>
/// at a specific epoch. It is awaitable, and yields either a reference or value
/// depending on value category:
///
/// - `co_await buf` yields `T const&`: shared read access.
/// - `co_await std::move(buf)` yields `T` for exclusive read access. The buffer
///   will attempt to move the value if it is the last reader; otherwise it will copy.
///
/// \note The `std::move(buf)` form is recommended when the buffer will be consumed
///       immediately, such as when assigning to a local variable.
///
/// \note A `ReadBuffer<T>` can be co_awaited multiple times, but `std::move(buf)`
///       transfers ownership semantics and should only be used once. After moving,
///       further use is undefined.
///
/// \tparam T The underlying value type.
// TODO: actually moving the value is not yet implemented
template <typename T> class ReadBuffer {
  public:
    /// \brief Construct a read buffer tied to a reader context.
    /// \param reader The RAII epoch reader handle for this operation.
    explicit ReadBuffer(EpochContextReader<T> reader) : reader_(std::move(reader)) {}

    ReadBuffer(ReadBuffer const&) = default;

    // No copy ctor here, although we could add one
    ReadBuffer& operator=(ReadBuffer const&) = delete;

    ReadBuffer(ReadBuffer&&) noexcept = default;
    ReadBuffer& operator=(ReadBuffer&&) noexcept = default;

    /// \brief Check if the value is already ready to be read.
    /// \return True if the epoch is ready and no suspension is needed.
    bool await_ready() const noexcept { return reader_.ready(); }

    /// \brief Suspend this coroutine and enqueue for resumption.
    /// \param t Coroutine task to enqueue.
    void await_suspend(AsyncTask&& t) noexcept { reader_.suspend(std::move(t)); }

    /// \brief Resume execution and return the stored value.
    /// \return Reference to the stored T inside Async<T>.
    T const& await_resume() const& noexcept { return reader_.data(); }

    /// \brief Resume execution and return a copy of the stored value.
    /// \note Called when co_awaiting on a prvalue ReadBuffer.
    T await_resume() && noexcept
    {
      static_assert(std::is_copy_constructible_v<T>, "Cannot co_await prvalue ReadBuffer<T> unless T is copyable");
      return reader_.data();
    }

    /// \brief Manually release the epoch reader before awaitable destruction.
    ///
    /// This allows the coroutine to relinquish its reader role earlier than
    /// its full lifetime.
    ///
    /// \post The ReadBuffer becomes inert and idempotent; calling `release()`
    ///       more than once has no effect.
    void release() noexcept { reader_.release(); }

    /// \brief Enable co_await on lvalue ReadBuffer only.
    ///
    /// Prevents unsafe use on temporaries by deleting rvalue overload.
    auto operator co_await() & noexcept -> ReadBuffer& { return *this; }
    auto operator co_await() const& noexcept -> ReadBuffer const& { return *this; }

    /// \brief Enable co_await on rvalue ReadBuffer. Returns by value.
    /// \note This avoids dangling reference when co_awaiting on a temporary.
    auto operator co_await() && noexcept -> ReadBuffer&& { return std::move(*this); }

  private:
    EpochContextReader<T> reader_; ///< RAII object managing epoch state.
};

/// \brief Awaitable write-gate for an Async<T> value.
///
/// Represents a single writer coroutine that attempts to gain exclusive
/// write access to an Async<T> value. Constructed from an EpochContextWriter<T>,
/// which manages ownership and coordination. This is a move-only awaiter that
/// binds once and either suspends or proceeds directly based on epoch ordering.
///
/// \note Not copyable. Must not be co_awaited on a temporary.
/// \note To pass a WriteBuffer into a nested coroutine, use dup() to retain ownership.
///       Be aware that duplicated WriteBuffers point to the same epoch — they do not
///       insert additional causality or memory ordering into the queue.
///
/// \warning Multiple active WriteBuffers to the same Async<T> are not causally isolated.
/// It is the user's responsibility to ensure they are not used concurrently or
/// inconsistently. This is akin to having multiple references to a shared global variable.
template <typename T> class WriteBuffer {
  public:
    /// \brief Construct a write buffer from an RAII writer handle.
    explicit WriteBuffer(EpochContextWriter<T> writer) : writer_(std::move(writer)) {}

    // Not copyable, but instead we have dup(WriteBuffer&) function
    WriteBuffer(WriteBuffer const&) = delete;
    WriteBuffer& operator=(WriteBuffer const&) = delete;

    WriteBuffer(WriteBuffer&&) noexcept = default;
    WriteBuffer& operator=(WriteBuffer&&) noexcept = default;

    /// \brief Check if this writer may proceed immediately.
    /// \return True if the epoch is at the front of the queue.
    bool await_ready() const noexcept { return writer_.ready(); }

    /// \brief Suspend the coroutine and bind as epoch writer.
    /// \param t Coroutine task to enqueue or bind.
    void await_suspend(AsyncTask&& t) noexcept { writer_.suspend(std::move(t)); }

    /// \brief Resume and return a reference to the writable value.
    /// \return Mutable reference to the stored T.
    T& await_resume() const noexcept { return writer_.data(); }

    /// \brief Manually release the write gate, allowing queue advancement.
    ///
    /// This may be called before destruction if the write is complete.
    /// It is safe and idempotent to call more than once.
    void release() noexcept { writer_.release(); }

    /// \brief Enable co_await on lvalue WriteBuffer only.
    ///
    /// Prevents unsafe use on temporaries by deleting rvalue overload.
    auto operator co_await() & noexcept -> WriteBuffer& { return *this; }
    auto operator co_await() const& noexcept -> WriteBuffer const& { return *this; }

    /// \brief Deleted rvalue co_await to avoid use-after-move or lifetime errors.
    auto operator co_await() && = delete;

  private:
    EpochContextWriter<T> writer_; ///< RAII handle for write coordination.

    /// \brief Duplicate a WriteBuffer to the same epoch and parent.
    ///
    /// This creates another WriteBuffer<T> referencing the same EpochContext and Async<T>.
    /// Use this when passing a WriteBuffer into another coroutine while retaining access in the caller.
    ///
    /// \note This does not alter the DAG — no new epoch is created. Both buffers refer to
    /// the same pending write. The caller must ensure only one write actually occurs, or
    /// they are otherwise syncronized.
    friend WriteBuffer dup(WriteBuffer& wb) { return WriteBuffer(wb.writer_); }
};

} // namespace uni20::async
