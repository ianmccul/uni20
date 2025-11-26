/// \file epoch_context.hpp
/// \brief Manages one “generation” of write/read ordering in an Async<T>
/// \ingroup async_core

#pragma once

#include "epoch_context_decl.hpp"
#include "epoch_queue.hpp"

namespace uni20::async
{

#if UNI20_DEBUG_DAG
template <typename T> inline NodeInfo const* EpochContextReader<T>::node() const
{
  return queue_ ? queue_->node() : nullptr;
}
#endif

template <typename T> inline void EpochContextReader<T>::suspend(AsyncTask&& t)
{
  TRACE_MODULE(ASYNC, "suspend", &t, epoch_.get());
  DEBUG_PRECONDITION(epoch_);
  queue_->enqueue_reader(epoch_, std::move(t));
}

template <typename T> inline bool EpochContextReader<T>::is_front() const noexcept
{
  DEBUG_PRECONDITION(epoch_, this);
  return queue_->is_front(epoch_.get());
}

template <typename T> inline void EpochContextReader<T>::release() noexcept
{
  if (epoch_)
  {
    if (epoch_->ctx.reader_release()) queue_->on_all_readers_released(epoch_);
    epoch_.reset();
  }
}

#if UNI20_DEBUG_DAG
template <typename T> inline NodeInfo const* EpochContextWriter<T>::node() const
{
  return queue_ ? queue_->node() : nullptr;
}
#endif

template <typename T> inline bool EpochContextWriter<T>::ready() const noexcept
{
  DEBUG_PRECONDITION(epoch_);
  return queue_->is_front(epoch_.get());
}

template <typename T> inline void EpochContextWriter<T>::suspend(AsyncTask&& t)
{
  DEBUG_PRECONDITION(epoch_);
  TRACE_MODULE(ASYNC, "suspend", &t, epoch_.get(), epoch_->ctx.counter_);
  epoch_->ctx.writer_bind(std::move(t));
  queue_->on_writer_bound(epoch_);
}

template <typename T> inline T& EpochContextWriter<T>::data() const noexcept
{
  DEBUG_PRECONDITION(storage_);                       // the value must exist
  DEBUG_PRECONDITION(queue_->is_front(epoch_.get())); // we must be at the front of the queue
  DEBUG_PRECONDITION(!epoch_->ctx.writer_is_done());  // writer still holds the gate
  accessed_ = true;
  auto* ptr = storage_->get();
  DEBUG_CHECK(ptr);
  return *ptr;
}

template <typename T> inline void EpochContextWriter<T>::release() noexcept
{
  if (epoch_)
  {
    if (accessed_ && !marked_written_)
    {
      epoch_->ctx.writer_has_written();
      marked_written_ = true;
    }
    if (epoch_->ctx.writer_release()) queue_->on_writer_done(epoch_);
    epoch_.reset();
    accessed_ = false;
    marked_written_ = false;
  }
}

} // namespace uni20::async
