#include "async/async.hpp"
#include "async/async_ops.hpp"
#include "async/awaiters.hpp"
#include "gtest/gtest.h"

using namespace uni20::async;

TEST(ConceptTest, AsyncIntSatisfiesConcepts)
{
  static_assert(async_reader<Async<int>>);
  static_assert(async_writer<Async<int>>);
  static_assert(async_like<Async<int>>);
}

TEST(ConceptTest, ReadBufferSatisfiesConcept) { static_assert(read_buffer_awaitable_of<ReadBuffer<int>, int>); }

TEST(ConceptTest, WriteBufferSatisfiesConcepts)
{
  static_assert(write_buffer_awaitable_of<WriteBuffer<int>, int>);
  static_assert(read_write_buffer_awaitable_of<WriteBuffer<int>, int>);
}

TEST(ConceptTest, MutableBufferSatisfiesConcepts)
{
  static_assert(write_buffer_awaitable_of<MutableBuffer<int>, int>);
  static_assert(read_write_buffer_awaitable_of<MutableBuffer<int>, int>);
}

TEST(ConceptTest, AsyncDoubleSatisfiesConcepts)
{
  static_assert(async_reader<Async<double>>);
  static_assert(async_writer<Async<double>>);
  static_assert(async_like<Async<double>>);
}
